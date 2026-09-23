# Kimi K3：上游 KDA 与官方卷积

分支 `fix/kimi-native-kda-conv-namespace` 基于官方 `releases/v0.26.0rc`
提交 `2ed6cbbaf481d84cfd3f2d01d47bcf79ee064c1f`，采用以下组合：

| 部分 | 使用的实现 |
| --- | --- |
| KDA Prefill | 上游 `kda_gate_cumsum`、`chunk_kda_fwd` |
| KDA Decode/Verify | 上游 `recurrent_kda` |
| Kimi ShortConv Prefill | CANN `cann_ops_transformer.causal_conv1d_fn` |
| Kimi ShortConv Decode/Verify | CANN `cann_ops_transformer.causal_conv1d_update` |
| 本仓 GDN 自定义卷积 | `_C_ascend.npu_causal_conv1d_custom`，底层改名为 `VllmCausalConv1d` |

本分支不引入 recipes 的 `flash_kda`、`fused_recurrent_kda_op` 或 `cannbot-dsl`
依赖。KDA 的 gate/beta 处理、状态布局、Prefill 和 recurrent 调用保持上游实现。
仅将 Kimi 的卷积调用适配到官方 CANN API：Prefill 保留已有缓存标记，普通 Decode
使用 `[B, 1, D]`，投机验证使用二维输入与真实接受数。

自定义卷积同步修改 OpDef、ACLNN 入口、kernel、tiling 和构建选择，计算逻辑和
Python 接口保持不变。官方 `CausalConv1d` 与本仓 `VllmCausalConv1d` 使用不同
底层注册名。Ascend 310P 的 `CausalConv1dV310` 保持不变。

## 在服务器切换并安装

先停止 vLLM 服务和 Ray workers，在每个节点加载所用 CANN 版本的 `set_env.sh`。
已有服务器 remote 名为 `wjcfork` 时，首次切换到新分支执行：

```bash
cd /data/w50063966/vllm-ascend
git fetch wjcfork
git switch --track -c fix/kimi-native-kda-conv-namespace \
    wjcfork/fix/kimi-native-kda-conv-namespace
MAX_JOBS=1 COMPILE_CUSTOM_KERNELS=1 \
    /usr/local/python3.11.10/bin/python3 -m pip install -v -e . \
    --no-build-isolation --no-deps
```

这会重新生成本仓算子包并编译 C++ 扩展。只有四个节点安装成功后，才能重启
Ray workers 和服务；仅切换 Git 分支不会更新已经编译的算子。
运行环境需要 CANN 提供可用的 `cann_ops_transformer` Fn/Update 接口，以及与
当前 CATLASS 源码兼容的编译器。本分支不包含 CATLASS 的额外修改。

## 验证范围

已补充官方卷积调用契约、旧注册检查和两套卷积共存的回归测试。
共存测试在独立进程中检查 FP16/BF16 的卷积结果、Prefill 续算及 Decode 缓存更新：

```bash
/usr/local/python3.11.10/bin/python3 -m pytest -v \
    tests/e2e/nightly/single_node/ops/singlecard_ops/test_causal_conv1d_coexistence.py
```

仅完成静态审查，未执行本地构建或 NPU 测试。服务器仍需验证完整服务启动、
首请求、连续 Decode，以及实际启用的 chunked prefill 和投机解码路径。
