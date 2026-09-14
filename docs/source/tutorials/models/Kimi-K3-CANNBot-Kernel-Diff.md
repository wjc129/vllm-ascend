# CANNBot KDA 算子源码差异说明

本文先记录 v0.26 引入 recipe 算子时的源码差异，再说明本分支的 v0.27 调用适配。v0.26 部分只比较以下两个算子文件：

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

## v0.26 引入时的总体结论

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

## v0.26 源码差异对功能和性能的影响

在没有设置 `KDA_GROUP`、`KDA_DV_BASE` 和 `KDA_ROW_BLOCK` 时，两个仓库都调用相同的自动选型函数，因此 kernel 配置、数值结果和预期性能应保持一致。

如果 recipe 的运行脚本显式设置了这些变量，差异才会出现：recipe 使用指定值，vllm-ascend 忽略指定值并继续自动选型。这可能改变编译出的 kernel 配置和性能，但不会改变 KDA 的数学定义。

以上结论来自源码逐行比较。本地没有 Atlas 950 环境，尚未进行 NPU 数值与性能验证。

## v0.27.0 分支的移植

本分支基于上游 `releases/v0.27.1rc` 的 `67dd255260e531e159c9de0103b70613ed140107`，迁移本仓库 v0.26 的 `0f7d8f4`、`fb872f9`、`fdd26f7` 三个提交。分支名为 `v0.27.0`，构建依赖继续沿用上游基线中的 vLLM `v0.27.1`。

### vLLM 调用适配

保留 v0.27 的 `KimiK3DeltaAttention` 基类、权重加载、混合精度投影、BFG 双流调度、`o_norm` 和 graph padding 输出清零逻辑。普通投影和混合精度 BFG 投影都传入原始 beta，由 CANNBot 内核进行 sigmoid，不再在 Python 层提前激活。

Prefill 使用 `flash_kda`。按请求拆分 packed token，将长度补到 batch 最长序列向上取整的 64 倍数，Q/K/V 补零、raw gate/beta 补负无穷；调用后去除 padding 并还原 token 顺序。初始状态以 FP32、`[H,V,K]` 布局传入，最终状态按原缓存 dtype 写回。

Decode/Verify 使用 `fused_recurrent_kda_op`。v0.27 的状态表保留最大草稿宽度，实际 query 可以更短。适配层按设备侧 `cu_seqlens` 将 packed 输入整理成 `[B,S,H,D]`，传入真实 `query_lengths`、原始状态槽索引和 `num_accepted_tokens`，最后还原 packed 输出。零长度 graph 行不更新缓存，padding 输出不会覆盖真实 token。

因果卷积分别调用 `cann_ops_transformer.causal_conv1d_fn` 和 `causal_conv1d_update`。编译依赖为 `ninja==1.13.0`、`cannbot-dsl`；两个 A5 Dockerfile 均包含安装步骤。算子路径保持为 `ops/cannbot_dsl/`，LICENSE 随包发布，该目录排除在 Ruff 检查之外。

对于 MRV2 的非连续卷积缓存，仅收集本批次请求的状态页到连续缓冲区，调用后写回原始 view；连续缓存直接传入。第三方卷积使用零号槽作为 null block，适配时保留这一约定，并把零长度 query 映射到 null block，避免 graph padding 行更新缓存。

### v0.27 的 recurrent 接口补充

`flash_kda.py` 保持与 v0.26 移植版本相同。`fused_recurrent_kda.py` 在原内核基础上增加以下调用兼容，KDA 的归一化、门控、状态更新公式和自动调优规则保持不变：

- 新增可选 `query_lengths`，按每个请求的真实长度执行，跳过零长度请求和无效状态槽；不传时保留原来的等长序列行为。`num_accepted_tokens` 仍在完整草稿状态表中选择初态，不按当前 query 长度裁剪。
- BSND 路径允许 MRV2 的状态缓存存在页间 stride，保留原始 view 原地写回；每个 `[H,V,K]` 状态块内部仍要求连续。编译参数与缓存键继续使用实际 shape 和 stride。

因此，前文“算子主体和接口均未修改”的历史结论仅适用于 v0.26 对 recipe 的引入，不能用来描述本分支新增的 recurrent 调用接口。

### 使用边界与验证

目标平台为 Atlas 950，当前内核要求 head dimension 为 128、Q/K/V/gate/beta 为 BF16，且必须提供 `gate_lower_bound`，取值范围为 `[-5, 0]`。此移植直接替换 KDA 执行路径，没有增加自动硬件分流或其他设备回退。

已有单元测试同步更新了 raw beta、变长 Prefill、短草稿 Verify、缓存写回和 graph padding 的接口覆盖。按本项目要求，未执行本地测试、构建或 NPU 验证；数值正确性、ACL graph 和性能需要在远程 Atlas 950 环境验证。
