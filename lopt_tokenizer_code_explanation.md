# `lopt_tokenizer.py` 代码讲解

## 1. 文件用途

[`lopt_tokenizer.py`](./vllm_ascend/tokenization/lopt_tokenizer.py) 实现的是 LoPT（长文本并行 Tokenization）：把超长 Prompt 按字符切成带重叠的片段，在线程池中并行分词，再利用重叠区域把各片段的 token 安全地拼接起来。

任何环节不安全或失败时，代码都会自动退回原来的串行 Tokenizer。

主入口是：

```python
LosslessParallelTokenizer.encode()
```

## 2. 为什么要切片并行

普通分词过程是：

```text
完整长文本 -> Tokenizer -> token IDs
```

LoPT 的处理方式是：

```text
完整长文本
  ↓
切成多个带重叠区域的文本块
  ↓
线程池并行分词
  ↓
通过 offset_mapping 找到相邻块的公共 token
  ↓
去重并拼接
  ↓
添加特殊 token、执行截断
  ↓
最终 token IDs
```

例如配置：

```python
chunk_chars = 100
overlap_chars = 20
```

文本会大致切成：

```text
chunk 0: text[0:120]
chunk 1: text[100:220]
chunk 2: text[200:320]
```

相邻块有 20 个字符重叠。重叠部分用于确定正确的拼接位置，避免直接切开一个 token。

## 3. 配置 `LoptConfig`

`LoptConfig` 定义了运行参数：

```python
@dataclass(frozen=True)
class LoptConfig:
    enabled: bool = False
    thread_workers: int = 4
    min_chars: int = 32768
    chunk_chars: int = 32768
    overlap_chars: int = 512
    min_match_tokens: int = 2
    max_retries: int = 3
    verify: bool = False
```

各参数含义：

- `enabled`：是否启用 LoPT。
- `thread_workers`：并行分词线程数。
- `min_chars`：文本至少达到多少字符才启用。
- `chunk_chars`：每个主体分片的字符数。
- `overlap_chars`：相邻片段重叠的字符数。
- `min_match_tokens`：拼接时至少找到多少个连续相同 token。
- `max_retries`：拼接失败后最多重试次数。
- `verify`：是否用普通分词结果做最终完整校验。

`frozen=True` 表示配置创建后不能修改。

`__post_init__()` 会检查参数是否合法，例如：

```python
if self.thread_workers < 1:
    raise ValueError(...)
```

## 4. 三个中间数据结构

### 4.1 `TextChunk`

```python
@dataclass(frozen=True)
class TextChunk:
    index: int
    global_start: int
    global_end: int
    text: str
```

表示一个文本分片：

- `index`：片段编号。
- `global_start`、`global_end`：该片段在原始文本中的字符位置。
- `text`：片段内容。

### 4.2 `ChunkEncoding`

```python
@dataclass(frozen=True)
class ChunkEncoding:
    index: int
    global_start: int
    global_end: int
    token_ids: tuple[int, ...]
    local_offsets: tuple[tuple[int, int], ...]
    global_offsets: tuple[tuple[int, int], ...]
```

表示一个片段分词后的结果：

- `token_ids`：分词得到的 token ID。
- `local_offsets`：token 在当前片段中的字符范围。
- `global_offsets`：token 在完整原始文本中的字符范围。

例如片段从原文第 1000 个字符开始，某个 token 的局部位置是：

```python
local_offset = (20, 24)
```

那么它的全局位置就是：

```python
global_offset = (1020, 1024)
```

### 4.3 `OverlapMatch`

`OverlapMatch` 保存相邻片段的匹配结果：

```python
@dataclass(frozen=True)
class OverlapMatch:
    left_start: int
    right_start: int
    token_count: int
    char_start: int
    char_end: int
```

它记录左右片段从哪个 token 开始重合、连续匹配了多少 token，以及匹配的字符范围。

## 5. 文本切片

`_split_overlapping()` 负责产生重叠片段：

```python
end = min(len(text), start + chunk_chars + overlap_chars)
...
start += chunk_chars
```

这里需要注意：

- 每个片段长度最多为 `chunk_chars + overlap_chars`。
- 下一片段只前进 `chunk_chars`。
- 所以相邻片段自然重叠 `overlap_chars` 个字符。

## 6. 单个片段分词

`_encode_chunk()` 调用 Hugging Face Fast Tokenizer：

```python
encoded = tokenizer(
    chunk.text,
    add_special_tokens=False,
    return_offsets_mapping=True,
    return_attention_mask=False,
    return_token_type_ids=False,
)
```

这里有两个关键点。

### 6.1 暂时不添加特殊 token

```python
add_special_tokens=False
```

否则每个片段都可能添加一次 BOS、EOS，拼接后会产生重复的特殊 token。特殊 token 会等所有片段合并后统一添加。

### 6.2 要求返回字符位置

```python
return_offsets_mapping=True
```

返回结果类似：

```python
input_ids = [100, 200, 300]
offset_mapping = [(0, 5), (5, 8), (8, 12)]
```

位置映射是后续安全拼接的核心。这也是 LoPT 只支持 Fast Tokenizer 的原因之一。

`_as_token_ids()` 和 `_as_offsets()` 会验证 Tokenizer 的返回值是否合法，包括：

- token ID 必须是整数。
- offset 必须是二元组。
- offset 不能超出文本范围。
- offset 必须单调递增。
- token 数量必须等于 offset 数量。

## 7. 并行分词

`_parallel_encode_chunks()` 将每个片段提交给线程池：

```python
futures = [
    executor.submit(_encode_chunk, tokenizer, chunk)
    for chunk in chunks
]
```

然后等待结果：

```python
encodings = [future.result() for future in futures]
```

因为并行任务的完成顺序不固定，最后按照片段编号重新排序：

```python
encodings.sort(key=lambda encoding: encoding.index)
```

只要一个片段失败，代码就尝试取消其他尚未执行的任务，并触发安全回退。

## 8. 如何寻找重叠位置

`_find_position_overlap()` 是核心算法。

它不是只比较 token ID，而是比较：

```python
(global_start, global_end, token_id)
```

也就是同时要求：

- token ID 相同。
- token 在原始文本中的起点相同。
- token 在原始文本中的终点相同。

例如：

```text
左片段：[(100,105,11), (105,110,22), (110,115,33)]
右片段：[(105,110,22), (110,115,33), (115,120,44)]
```

可以找到公共序列：

```text
(105,110,22), (110,115,33)
```

这种方式比单纯比较 token ID 更安全，因为文本中可能重复出现相同 token。

当存在多个候选匹配时，代码会按照以下条件排序：

```python
return match.token_count, safety_margin, character_span
```

优先选择：

1. 连续匹配 token 更多的。
2. 距离重叠区边缘更远的。
3. 覆盖字符范围更大的。

如果前两个候选排名完全一样，代码认为匹配存在歧义，返回 `None`，而不是冒险拼接。

## 9. 合并片段

`_merge_pair()` 找到公共 token 后执行：

```python
token_ids = left.token_ids[:left_end] + right.token_ids[right_end:]
```

含义是：

```text
左片段匹配区域之前及匹配区域
+
右片段匹配区域之后的内容
```

这样公共区域只保留一份。

`_merge_chunk_encodings()` 则按照顺序不断两两合并：

```python
merged = encodings[0]
for encoding in encodings[1:]:
    merged = _merge_pair(merged, encoding, min_match_tokens)
```

## 10. 什么时候启用 LoPT

`can_use()` 会逐项检查：

```python
if is_shutdown or not self.config.enabled:
    return False
if tokenizer.is_fast is not True:
    return False
if len(text) < self.config.min_chars:
    return False
if len(text) <= chunk_chars + overlap_chars:
    return False
```

只有同时满足以下条件才使用并行分词：

- LoPT 已启用。
- 线程池没有关闭。
- 使用 Fast Tokenizer。
- 文本足够长。
- 文本能切成至少两个片段。
- 参数仅包含支持的参数。
- Tokenizer 接口满足要求。
- 兼容性探测成功。

目前只支持这些 `encode()` 参数：

```python
{"add_special_tokens", "truncation", "max_length"}
```

传入其他参数时，会退回普通分词。

## 11. 兼容性探测

`_is_compatible()` 会构造一个包含 ASCII、Unicode、组合字符和数字的短文本，然后比较：

```text
LoPT 分词并补充特殊 token 后的结果
                是否等于
Tokenizer 直接 encode 的结果
```

结果会按照下面的配置组合缓存：

```python
(add_special_tokens, truncation, max_length)
```

如果探测失败，该组合后续就不再使用 LoPT。

## 12. 主入口 `encode()`

核心逻辑可以概括为：

```python
def encode(self, text, **encode_kwargs):
    if not self.can_use(text, encode_kwargs):
        return self._standard_encode(text, encode_kwargs)

    try:
        token_ids = 并行切片、分词、拼接
        if verify:
            standard_ids = 普通分词
            if token_ids != standard_ids:
                return standard_ids
    except Exception:
        return self._standard_encode(text, encode_kwargs)

    return token_ids
```

下面这段：

```python
return self._standard_encode(text, encode_kwargs)
```

表示当前文本或 Tokenizer 不适合使用 LoPT，因此直接调用原始 Tokenizer：

```python
def _standard_encode(self, text, encode_kwargs):
    return list(self.tokenizer.encode(text, **encode_kwargs))
```

成功走完并行流程后，下面的日志代码：

```python
logger.debug(...)
```

只记录以下性能信息，不会改变分词结果：

- 输入字符数量。
- 输出 token 数量。
- 使用的片段数量。
- 重试次数。
- 总耗时。

## 13. 失败重试

`_encode_with_retries()` 在重叠匹配失败时，会将片段扩大一倍：

```python
chunk_chars *= 2
```

然后重新切片、并行分词和匹配。

这样切分边界会发生变化，同时片段数量减少，可能避开难以匹配的边界。

超过 `max_retries` 后仍失败，就抛出异常；外层 `encode()` 捕获异常并退回普通分词。

## 14. 添加特殊 token 和截断

`_prepare_for_model()` 在所有普通 token 合并完成后，再统一处理：

- BOS、EOS 等特殊 token。
- `truncation`。
- `max_length`。

优先调用 Hugging Face Tokenizer 的：

```python
tokenizer.prepare_for_model(...)
```

如果没有该方法，并且只需要特殊 token，则尝试：

```python
tokenizer.build_inputs_with_special_tokens(...)
```

如果无法准确重现用户要求，就抛出异常并退回标准分词。

## 15. 它是怎么接入 vLLM 的

这个类本身不会主动运行。[`patch_lopt_tokenization.py`](./vllm_ascend/patch/platform/patch_lopt_tokenization.py) 会修改 `HfRenderer`。

初始化 Renderer 时创建 LoPT Tokenizer：

```python
self._ascend_lopt_tokenizer = LosslessParallelTokenizer(
    tokenizer,
    _config_from_env(),
)
```

处理 Prompt 时：

```python
if lopt is None or not lopt.can_use(text, encode_kwargs):
    return 原始分词逻辑

prompt_token_ids = lopt.encode(text, **encode_kwargs)
```

完整调用链如下：

```text
VLLM_ASCEND_LOPT_ENABLE=true
  ↓
导入 patch.platform.patch_lopt_tokenization
  ↓
修改 HfRenderer 的分词方法
  ↓
收到长 Prompt
  ↓
LosslessParallelTokenizer.can_use()
  ├─ 不满足条件 -> 原始 Tokenizer
  └─ 满足条件
       ↓
     重叠切片
       ↓
     线程池并行分词
       ↓
     按全局字符位置匹配、去重、合并
       ↓
     添加特殊 token/截断
       ↓
     成功返回；任何异常则回退原始 Tokenizer
```

## 16. 关于 “Lossless”

这里的 “Lossless” 更准确地说是设计目标。代码通过以下机制保证安全：

- 使用字符位置和 token ID 共同匹配。
- 匹配存在歧义时拒绝拼接。
- 执行兼容性探测。
- 发生异常时退回原始 Tokenizer。
- 可选择执行完整结果校验。

当 `verify=True` 时，代码还会完整运行一次普通分词并逐项比较。只有并行分词结果与普通分词结果完全一致时，才采用并行结果；否则返回普通分词结果。
