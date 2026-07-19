# LoPT 并行 Tokenization 改动说明

## 概述

本次改动在 vLLM Ascend 中增加了实验性的 LoPT（Lossless Overlapping Parallel Tokenization）能力，用于降低单个超长文本 Prompt 的 CPU tokenization 延迟。

标准 Tokenizer 会一次性串行处理完整 Prompt。LoPT 则将文本切分为带重叠区域的多个 Chunk，使用线程池并行编码，再根据 token ID 和全局字符位置合并各个结果。合并完成后，代码统一执行特殊 token 添加和截断，最终输出与标准 vLLM 路径格式一致的 token IDs。

LoPT 位于请求进入 EngineCore 之前的 CPU 预处理阶段，不改变模型结构、权重加载、Scheduler、KV Cache 或 NPU 计算流程。该功能默认关闭，仅在设置 `VLLM_ASCEND_LOPT_ENABLE=1` 时加载。

当前实现具备三个基本特征：

- 只处理文本模型使用的 Hugging Face Fast Tokenizer；
- 无法确认结果安全时自动回退到标准 Tokenizer；
- 可通过 `VERIFY` 模式与标准分词结果进行完整比对。

## 整体流程

LoPT 从插件初始化到请求处理的调用链如下：

```text
启动 vLLM Ascend
  -> 读取 VLLM_ASCEND_LOPT_* 环境变量
  -> 设置 Tokenizer 内部线程默认值
  -> 加载 HfRenderer 平台补丁
  -> 为文本 Fast Tokenizer 创建 LoPT 线程池
  -> 接收 Completion Prompt 或展开后的 Chat Prompt
  -> 检查文本长度、Tokenizer 类型、编码参数和兼容性
      -> 不满足条件：调用原始 HfRenderer 分词逻辑
      -> 满足条件：进入 LoPT
          -> 生成重叠文本块
          -> 并行执行无特殊 token 的分词
          -> 将局部 offset 转换为全局 offset
          -> 匹配并合并相邻文本块
          -> 统一执行特殊 token 和截断
          -> 可选地与标准结果完整比对
          -> 返回 token IDs
  -> 任何异常或歧义均回退到标准 Tokenizer
```

LoPT 的结果仍然以 `TokensPrompt` 形式交给 vLLM，后续调度和 NPU 推理流程不需要感知当前请求采用了哪一种 tokenization 路径。

## 核心实现

### 重叠切片与并行编码

核心逻辑位于 `vllm_ascend/tokenization/lopt_tokenizer.py`。其中，`LoptConfig` 保存线程数、启用阈值、Chunk 长度、重叠长度、匹配要求、重试次数和校验开关。配置对象使用 `frozen=True`，创建后不可修改，并在 `__post_init__()` 中检查数值范围。

`_split_overlapping()` 按字符位置生成 Chunk。每个 Chunk 的最大长度为 `chunk_chars + overlap_chars`，下一片的起点向后移动 `chunk_chars`：

```text
chunk_chars = 100
overlap_chars = 20

Chunk 0: text[0:120]
Chunk 1: text[100:220]
Chunk 2: text[200:320]
```

相邻 Chunk 共享 20 个字符。这部分重复文本为边界匹配提供上下文，合并后只保留一份。

每个 Chunk 通过 Hugging Face Fast Tokenizer 独立编码：

```python
tokenizer(
    chunk.text,
    add_special_tokens=False,
    return_offsets_mapping=True,
    return_attention_mask=False,
    return_token_type_ids=False,
)
```

分片阶段暂时关闭特殊 token，避免每个 Chunk 重复产生 BOS 或 EOS。代码同时要求 Tokenizer 返回 `offset_mapping`，并检查 token IDs 是否为一维整数序列、offset 是否合法且单调、offset 数量是否与 token 数量一致。

Tokenizer 返回的 offset 最初以当前 Chunk 为基准。LoPT 将其与 Chunk 在原始文本中的起点相加，转换为全局字符位置。例如，Chunk 从原文字符 1000 开始，局部 offset 为 `(20, 24)`，对应的全局 offset 就是 `(1020, 1024)`。

`LosslessParallelTokenizer` 持有一个长期存活的 `ThreadPoolExecutor`。所有 Chunk 被提交到线程池并行处理，任务完成后再按 Chunk 序号恢复原始顺序。任何一个任务失败时，代码会尝试取消尚未开始的任务，并由外层逻辑切换到标准分词。

### 全局位置匹配与结果合并

LoPT 不会只根据 token ID 判断两个 Chunk 是否重叠。相同 token 可能在 Prompt 中重复出现，仅比较 ID 容易选择错误的拼接位置。

匹配阶段将每个 token 表示为：

```text
(global_start, global_end, token_id)
```

左右两侧的全局起点、全局终点和 token ID 必须同时一致，才能视为同一个 token。代码在相邻 Chunk 的字符重叠区内寻找连续相同的记录，并要求连续长度达到 `min_match_tokens`。

存在多个候选时，匹配器优先选择连续 token 更多、距离重叠区边缘更远、覆盖字符范围更大的序列。如果两个最佳候选的排名完全相同，结果会被视为存在歧义，LoPT 不会任选一个位置继续拼接。

确定公共序列后，合并逻辑保留左侧内容和一份公共序列，再追加右侧公共序列之后的内容：

```python
token_ids = left.token_ids[:left_end] + right.token_ids[right_end:]
```

全局 offsets 使用相同边界合并，确保 token IDs 和字符位置始终保持同步。所有 Chunk 按顺序两两合并，最终形成完整 Prompt 的内容 token 序列。

### 后处理、重试与结果校验

内容 token 合并完成后，`_prepare_for_model()` 统一处理 `add_special_tokens`、`truncation` 和 `max_length`。代码优先调用 Tokenizer 的 `prepare_for_model()`，复用原有后处理规则；接口不足以准确重现请求语义时，当前请求直接回退。

部分 Tokenizer 的编码结果会受到切片边界影响。重叠匹配失败后，`_encode_with_retries()` 会将 `chunk_chars` 扩大一倍，减少切片数量并改变边界位置，然后重新执行切片、编码和合并。超过 `max_retries` 后仍无法建立唯一匹配，则返回标准 Tokenizer 的结果。

Fast Tokenizer 也不一定保证“先编码内容 token，再执行 `prepare_for_model()`”与标准 `encode()` 完全一致。因此，LoPT 会使用包含 ASCII、中文、Emoji、组合字符和数字的短文本执行兼容性探测，并按 `(add_special_tokens, truncation, max_length)` 缓存结果。探测失败的参数组合后续持续使用标准路径。

启用 `VLLM_ASCEND_LOPT_VERIFY=1` 后，满足 LoPT 条件的长 Prompt 还会完整执行一次标准分词。两组 token IDs 完全一致时才保留 LoPT 结果，否则返回标准结果。该模式适合 Tokenizer 准入验证，但会抵消并行分词的性能收益。

## `HfRenderer` 集成与线程管理

`vllm_ascend/patch/platform/patch_lopt_tokenization.py` 保存 `HfRenderer` 的原始实现，并替换初始化、关闭、Chat 渲染以及同步和异步 Prompt tokenization 入口。补丁通过 `_PATCH_APPLIED` 保证同一进程只应用一次。

Renderer 初始化完成后，补丁只为“Tokenizer 存在、没有多模态 Processor、Tokenizer 为 Fast Tokenizer”的实例创建 `LosslessParallelTokenizer`。初始化异常只记录 warning，不影响 Renderer 使用原始路径。

对于已经创建 LoPT 实例的文本 Renderer，补丁会在调用原始 `HfRenderer.render_messages()` 前，将 `chat_template_kwargs["tokenize"]` 设置为 `False`，使 Chat Template 输出格式化文本。该文本随后进入统一的 Prompt tokenization 入口：满足 LoPT 条件时执行并行分词，否则调用原始 `HfRenderer` 分词逻辑。

同步入口直接调用 `lopt.encode()`。异步入口先通过 Renderer 自身的 executor 调度整个编码过程，避免阻塞事件循环；LoPT 内部再由专用线程池并行执行各个 Chunk。Renderer 关闭时，补丁先关闭 LoPT 线程池，再调用原始 `shutdown()`，避免遗留工作线程。

Hugging Face Fast Tokenizer 底层可能使用 Rayon。LoPT 已经在外层并行处理 Chunk，如果每个任务内部继续启动多个 Rayon 线程，会形成嵌套并行并导致 CPU 过度订阅。因此启用 LoPT 时，插件会使用 `setdefault()` 设置：

```text
RAYON_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

`setdefault()` 不会覆盖用户已经显式配置的值。相关设置同时位于 `vllm_ascend/__init__.py` 和补丁模块：前者保证尽早生效，后者覆盖直接导入补丁模块的路径。

所有 LoPT 异常都被视为优化失败，而不是请求失败。Prompt 太短、Slow Tokenizer、多模态 Renderer、不支持的编码参数、非法 offset、匹配歧义、线程任务异常或完整校验不一致，最终都会使用标准 Tokenizer。同类回退第一次记录 warning，后续降为 debug，避免重复刷屏。

## 配置与适用范围

LoPT 的环境变量集中定义在 `vllm_ascend/envs.py`：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `VLLM_ASCEND_LOPT_ENABLE` | `0` | LoPT 总开关 |
| `VLLM_ASCEND_LOPT_THREAD_WORKERS` | `4` | 每个 Renderer 的 Chunk 编码线程数 |
| `VLLM_ASCEND_LOPT_MIN_CHARS` | `32768` | 进入 LoPT 的最小 Unicode 字符数 |
| `VLLM_ASCEND_LOPT_CHUNK_CHARS` | `32768` | 每个 Chunk 的主体字符数 |
| `VLLM_ASCEND_LOPT_OVERLAP_CHARS` | `512` | 相邻 Chunk 的重叠字符数 |
| `VLLM_ASCEND_LOPT_MIN_MATCH_TOKENS` | `2` | 合并所需的最小连续匹配 token 数 |
| `VLLM_ASCEND_LOPT_MAX_RETRIES` | `3` | 匹配失败后的最大重试次数 |
| `VLLM_ASCEND_LOPT_VERIFY` | `0` | 是否与标准分词结果完整比较 |

长度配置使用 Python Unicode 字符数，不代表字节数或 token 数。默认参数用于提供保守起点，实际收益取决于 CPU 核数、Tokenizer 实现和 Prompt 长度分布。

| 能力 | 状态 | 说明 |
|---|---|---|
| 文本 Completion Prompt | 支持 | 满足长度和兼容性条件时进入 LoPT |
| Chat Prompt | 支持 | Chat Template 展开为文本后进入统一分词入口 |
| 同步与异步 tokenization | 支持 | 异步入口不会直接阻塞事件循环 |
| Hugging Face Fast Tokenizer | 条件支持 | 必须返回合法 offset 并通过兼容性探测 |
| Slow Tokenizer | 不支持 | 自动使用标准路径 |
| 多模态 Processor | 不支持 | 不创建 LoPT，不改变原始多模态流程 |
| 特殊 token 与截断 | 条件支持 | 依赖 Tokenizer 的标准后处理接口 |
| 额外的 `encode()` 参数 | 不支持 | 当前仅接受 `add_special_tokens`、`truncation` 和 `max_length` |
| ACLGraph、EP、MTP、FlashComm | 无直接关系 | LoPT 位于 NPU 执行流程之前 |

## 代码结构

| 文件 | 内容 |
|---|---|
| `vllm_ascend/tokenization/lopt_tokenizer.py` | LoPT 配置、数据结构、切片、编码、匹配、合并、重试和回退 |
| `vllm_ascend/tokenization/__init__.py` | LoPT 公共类型导出 |
| `vllm_ascend/patch/platform/patch_lopt_tokenization.py` | `HfRenderer` 集成 |
| `vllm_ascend/patch/platform/__init__.py` | 按环境变量加载补丁 |
| `vllm_ascend/__init__.py` | Tokenizer 内部线程默认值设置 |
| `vllm_ascend/envs.py` | LoPT 环境变量定义 |
| `vllm_ascend/patch/__init__.py` | 平台补丁登记和上游化说明 |
| `tests/ut/tokenization/test_lopt_tokenizer.py` | 核心算法和回退路径的单元测试 |
| `docs/source/user_guide/feature_guide/lopt_tokenization.md` | 用户功能说明 |
| `lopt_long_prompt_flowchart.md` | 完整请求流程图 |

核心代码可以从 `patch_lopt_tokenization.py` 的 Renderer 接入点开始阅读，再进入 `LosslessParallelTokenizer.encode()`，沿着切片、并行编码、重叠匹配和后处理流程展开。更细的函数级说明见 `lopt_tokenizer_code_explanation.md`。
