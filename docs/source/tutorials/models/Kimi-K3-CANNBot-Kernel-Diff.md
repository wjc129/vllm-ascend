# CANNBot KDA 算子源码差异说明

本文只比较以下两个算子文件，不讨论 `kimi_kda.py` 中的 vLLM 调用适配：

1. `flash_kda.py`：Prefill KDA
2. `fused_recurrent_kda.py`：Decode/Verify KDA

对比基线：

- recipe：`cann-recipes-infer` 提交 `803c3120f483d3ae8d9f73d6d8941de25a2863d7`
- vLLM：`vllm-ascend` 提交 `fb872f96da91de07edbfe22d7d16ccd4d29c6519`

对应文件：

| 算子 | recipe 原始版本 | vllm-ascend 版本 |
| --- | --- | --- |
| Prefill | `cann-recipes-infer/ops/cannbot_dsl/flash_kda.py` | `vllm-ascend/ops/cannbot_dsl/flash_kda.py` |
| Decode/Verify | `cann-recipes-infer/ops/cannbot_dsl/fused_recurrent_kda.py` | `vllm-ascend/ops/cannbot_dsl/fused_recurrent_kda.py` |

## 总体结论

vllm-ascend 中的两个文件来源于 recipe。CANNBot DSL kernel 主体、数学公式、tensor 布局、数据类型、JIT 编译和算子接口均未修改。

源码差异只涉及调优参数的环境变量覆盖和一个 PyTorch 自动加载环境设置：

| 文件 | 实际变化 |
| --- | --- |
| `flash_kda.py` | 删除 `TORCH_DEVICE_BACKEND_AUTOLOAD` 设置；删除 `KDA_GROUP`、`KDA_DV_BASE` 手工覆盖 |
| `fused_recurrent_kda.py` | 删除 `KDA_ROW_BLOCK` 手工覆盖；同步修改报错文字 |

未设置这些环境变量时，recipe 与 vllm-ascend 会选择相同的自动配置，算子行为没有变化。

## flash_kda.py 的差异

### 删除进程级环境设置

recipe 原始版本：

```python
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
```

vllm-ascend 删除了以上代码。算子模块不再在 import 阶段修改整个 Python 进程的 PyTorch backend 自动加载行为。这个变化不涉及 KDA 计算。

### group 只使用自动选型

recipe 原始版本：

```python
group = int(os.environ.get("KDA_GROUP") or get_group_config_bn(bn_total))
```

vllm-ascend 版本：

```python
group = get_group_config_bn(bn_total)
```

recipe 可通过 `KDA_GROUP` 强制指定 chunk group。vllm-ascend 始终根据 batch 与 head 数自动选择。未设置 `KDA_GROUP` 时，两者完全相同。

### dv_base 只使用自动选型

recipe 原始版本：

```python
dv_base = int(
    os.environ.get("KDA_DV_BASE")
    or get_dv_base_config(bn_total, seq_len)
)
```

vllm-ascend 版本：

```python
dv_base = get_dv_base_config(bn_total, seq_len)
```

recipe 可通过 `KDA_DV_BASE` 强制指定 value 维度切分。vllm-ascend 始终根据 batch、head 和序列长度自动选择。未设置该变量时，两者结果相同。

除以上三处外，`flash_kda.py` 没有其他有效代码差异。

## fused_recurrent_kda.py 的差异

### row_block 只使用自动选型

recipe 原始版本：

```python
row_block_env = os.environ.get("KDA_ROW_BLOCK")
row_block = (
    int(row_block_env)
    if row_block_env
    else get_row_block_config(batch * num_value_heads, seq_len)
)
```

vllm-ascend 版本：

```python
row_block = get_row_block_config(
    batch * num_value_heads,
    seq_len,
)
```

recipe 可通过 `KDA_ROW_BLOCK` 强制指定 recurrent kernel 的行分块。vllm-ascend 始终使用自动选型。未设置该变量时，两者得到相同的 `row_block`。

因为不再读取环境变量，文件顶部的 `import os` 也被删除。

### 断言信息调整

recipe 原始版本：

```python
f"KDA_ROW_BLOCK must divide D and be one of 16, 32, 64, got {row_block}"
```

vllm-ascend 版本：

```python
f"row_block must divide D and be one of 16, 32, 64, got {row_block}"
```

这里只修改异常文字，不影响执行逻辑。

除以上内容外，`fused_recurrent_kda.py` 没有其他有效代码差异。

## 对功能和性能的影响

在没有设置 `KDA_GROUP`、`KDA_DV_BASE` 和 `KDA_ROW_BLOCK` 时，两个仓库都调用相同的自动选型函数，因此 kernel 配置、数值结果和预期性能应保持一致。

如果 recipe 的运行脚本显式设置了这些变量，差异才会出现：recipe 使用指定值，vllm-ascend 忽略指定值并继续自动选型。这可能改变编译出的 kernel 配置和性能，但不会改变 KDA 的数学定义。

以上结论来自源码逐行比较。本地没有 Atlas 950 环境，尚未进行 NPU 数值与性能验证。
