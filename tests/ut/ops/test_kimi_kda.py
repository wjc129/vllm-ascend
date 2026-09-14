# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDynamicLinearMethod,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
)

# Exercise the adapter contract without loading the Atlas 950 compiler or
# native convolution library. Restore the dependency modules after import so
# other tests can import their real implementations independently.
with pytest.MonkeyPatch.context() as _kernel_modules:
    for _module_name in (
        "cann_ops_transformer",
        "cann_ops_transformer.ops",
        "ops.cannbot_dsl.flash_kda",
        "ops.cannbot_dsl.fused_recurrent_kda",
    ):
        _kernel_modules.setitem(sys.modules, _module_name, MagicMock())
    from vllm_ascend.ops.kimi_kda import (
        _PACKED_CONV_WEIGHT_NAME,
        AscendKimiK3DeltaAttention,
        _KDAFusedBFGLinear,
        _zero_padded_output,
        _zero_padded_recurrent_output,
    )


class _RecordingLinear(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, _input: torch.Tensor):
        return self.output, None


class _RecordingStream:
    def __init__(self, name: str, event_names: list[str], trace: list[str]) -> None:
        self.name = name
        self.event_names = iter(event_names)
        self.trace = trace

    def record_event(self) -> str:
        event = next(self.event_names)
        self.trace.append(f"{self.name}.record:{event}")
        return event

    def wait_event(self, event: str) -> None:
        self.trace.append(f"{self.name}.wait:{event}")


class _RecordingTensor:
    def __init__(self, name: str, trace: list[str]) -> None:
        self.name = name
        self.trace = trace

    def record_stream(self, stream: _RecordingStream) -> None:
        self.trace.append(f"{self.name}.record_stream:{stream.name}")


class _RecordingStreamSwitch:
    def __init__(self, stream: _RecordingStream, trace: list[str]) -> None:
        self.stream = stream
        self.trace = trace

    def __enter__(self) -> None:
        self.trace.append(f"enter:{self.stream.name}")

    def __exit__(self, *args) -> None:
        self.trace.append(f"exit:{self.stream.name}")


def test_zero_padded_recurrent_output_clears_uncovered_tail():
    output = torch.randn(1, 8, 2, 3)
    expected = output[:, :5].clone()
    output[:, 5:] = torch.nan

    actual = _zero_padded_recurrent_output(
        output,
        torch.tensor([0, 3, 5, 5], dtype=torch.int32),
    )

    torch.testing.assert_close(actual[:, :5], expected)
    assert torch.equal(actual[:, 5:], torch.zeros_like(actual[:, 5:]))
    assert torch.isfinite(actual).all()


def test_zero_padded_output_uses_combined_live_token_count():
    output = torch.full((1, 8, 1, 1), torch.nan)
    output[:, :6] = torch.arange(6).view(1, 6, 1, 1)

    actual = _zero_padded_output(output, torch.tensor(6, dtype=torch.int32))

    torch.testing.assert_close(actual[:, :6], output[:, :6])
    assert torch.equal(actual[:, 6:], torch.zeros_like(actual[:, 6:]))


def test_kda_output_norm_uses_checkpoint_epsilon():
    def fake_upstream_init(attention, _config, _vllm_config, _prefix):
        nn.Module.__init__(attention)
        attention.o_norm = SimpleNamespace(eps=1e-5)
        attention.conv_size = 4
        attention.local_projection_size = 2
        attention.gate_lower_bound = -5.0
        attention.model_config = SimpleNamespace(dtype=torch.bfloat16)
        attention.conv1d = nn.Module()
        attention.conv1d.weight = nn.Parameter(torch.empty(6, 1, 4))
        attention.conv1d.quant_method = SimpleNamespace(process_weights_after_loading=lambda: None)

    config = SimpleNamespace(rms_norm_eps=1e-6)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            multimodal_config=None,
            enable_prompt_embeds=False,
        )
    )
    with patch(
        "vllm_ascend.ops.kimi_kda.KimiK3DeltaAttention.__init__",
        new=fake_upstream_init,
    ):
        attention = AscendKimiK3DeltaAttention(config, vllm_config)

    assert attention.o_norm.eps == config.rms_norm_eps


@pytest.mark.parametrize("f_b_is_local", [False, True])
def test_fused_bfg_linear_composes_f_and_packs_bfg(f_b_is_local: bool):
    with (
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=4),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=2),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=2),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=4),
    ):
        linear = _KDAFusedBFGLinear(
            hidden_size=6,
            num_heads=8,
            head_dim=3,
            tp_size=4,
            quant_config=None,
            prefix="model.layers.0.self_attn.in_proj_gfab",
        )

    linear.weight.data.zero_()
    b_weight = torch.arange(8 * 6, dtype=linear.weight.dtype).reshape(8, 6)
    f_a_weight = torch.arange(3 * 6, dtype=linear.weight.dtype).reshape(3, 6) + 100
    global_f_b_weight = torch.arange(24 * 3, dtype=linear.weight.dtype).reshape(24, 3) + 200
    local_f_b_weight = global_f_b_weight[12:18]
    g_weight = torch.arange(24 * 6, dtype=linear.weight.dtype).reshape(24, 6) + 200

    linear.weight.weight_loader(linear.weight, b_weight, 0)
    linear.f_a_weight.weight_loader(linear.f_a_weight, f_a_weight)
    linear.f_b_weight.weight_loader(
        linear.f_b_weight,
        local_f_b_weight if f_b_is_local else global_f_b_weight,
    )
    linear.weight.weight_loader(linear.weight, g_weight, 2)

    expected_f = (local_f_b_weight.float() @ f_a_weight.float()).to(linear.weight.dtype)
    assert tuple(linear.weight.shape) == (14, 6)
    torch.testing.assert_close(linear.weight[:2], b_weight[4:6])
    torch.testing.assert_close(linear.weight[2:8], expected_f)
    torch.testing.assert_close(linear.weight[8:], g_weight[12:18])


def test_fused_bfg_linear_recomposes_f_after_source_reload():
    with (
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
    ):
        linear = _KDAFusedBFGLinear(
            hidden_size=4,
            num_heads=2,
            head_dim=2,
            tp_size=1,
            quant_config=None,
            prefix="model.layers.0.self_attn.in_proj_gfab",
        )

    linear.weight.data.zero_()
    first_f_a = torch.arange(8, dtype=linear.weight.dtype).reshape(2, 4)
    first_f_b = torch.arange(8, dtype=linear.weight.dtype).reshape(4, 2)
    linear.f_a_weight.weight_loader(linear.f_a_weight, first_f_a)
    torch.testing.assert_close(linear.weight[2:6], torch.zeros_like(linear.weight[2:6]))
    linear.f_b_weight.weight_loader(linear.f_b_weight, first_f_b)
    torch.testing.assert_close(linear.weight[2:6], first_f_b.float() @ first_f_a.float())

    reloaded_f_a = first_f_a + 10
    reloaded_f_b = first_f_b + 20
    linear.f_a_weight.weight_loader(linear.f_a_weight, reloaded_f_a)
    linear.f_b_weight.weight_loader(linear.f_b_weight, reloaded_f_b)
    torch.testing.assert_close(linear.weight[2:6], reloaded_f_b.float() @ reloaded_f_a.float())


def test_fused_bfg_projection_preserves_staged_outputs():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.head_dim = 3
    attention._fused_bfg_output_sizes = (2, 6, 6)
    hidden_states = torch.randn(4, 5)
    fused_output = torch.arange(56, dtype=torch.float32).reshape(4, 14).to(torch.bfloat16)
    attention.fused_bfg_proj = _RecordingLinear(fused_output)

    projected_bfg = attention._project_bfg(hidden_states)
    assert projected_bfg is fused_output

    beta, raw_gate, output_gate = attention._postprocess_bfg(projected_bfg)
    assert beta.dtype == torch.bfloat16
    torch.testing.assert_close(beta, fused_output[:, :2].unsqueeze(0))
    torch.testing.assert_close(raw_gate, fused_output[:, 2:8].reshape(4, 2, 3).unsqueeze(0))
    torch.testing.assert_close(output_gate, fused_output[:, 8:].reshape(4, 2, 3))


def test_mixed_forward_passes_raw_auxiliary_beta():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.uses_mixed_projection = True
    attention.local_num_heads = 2
    attention.head_dim = 3
    hidden_states = torch.randn(4, 6)
    positions = torch.arange(4)
    mixed_qkv = torch.randn(4, 18)
    beta = torch.tensor([-20.0, 20.0], dtype=torch.bfloat16).expand(1, 4, 2)
    raw_gate = torch.randn(1, 4, 2, 3)
    output_gate = torch.randn(4, 2, 3)
    projected = torch.randn(4, 6)
    attention._run_overlapped_qkv_bfg = MagicMock(return_value=(mixed_qkv, beta, raw_gate, output_gate))
    attention._forward = MagicMock()
    attention.o_proj = _RecordingLinear(projected)

    actual = attention.forward(hidden_states, positions)

    assert actual is projected
    assert attention._forward.call_args.kwargs["beta"] is beta


def test_forward_slices_raw_beta_and_clears_graph_padding():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.prefix = "model.layers.0.self_attn"
    attention.local_num_heads = 1
    attention.head_dim = 2
    attention.kv_cache = (torch.empty(4, 6, 3), torch.empty(4, 1, 2, 2))
    attention.register_parameter(_PACKED_CONV_WEIGHT_NAME, nn.Parameter(torch.empty(4, 6)))
    attention._run_causal_conv1d = MagicMock(side_effect=lambda x, *_args, **_kwargs: x)
    attention._run_recurrent = MagicMock(side_effect=lambda q, *_args, **_kwargs: q)
    attention.o_norm = MagicMock(side_effect=lambda output, _gate: output)
    metadata = MagicMock(spec=GDNAttentionMetadata)
    metadata.num_actual_tokens = 3
    metadata.spec_sequence_masks = None
    metadata.spec_token_indx = None
    metadata.non_spec_token_indx = None
    metadata.num_prefills = 0
    metadata.num_decodes = 2
    metadata.num_decode_tokens = 2
    metadata.non_spec_query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    metadata.non_spec_state_indices_tensor = torch.tensor([3, 1, -1], dtype=torch.int32)
    metadata.non_spec_decode_metadata = SimpleNamespace(
        causal_conv1d=SimpleNamespace(
            query_start_loc=metadata.non_spec_query_start_loc,
            cache_indices=metadata.non_spec_state_indices_tensor,
        )
    )
    mixed_qkv = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    raw_beta = torch.tensor([[[-20.0], [0.0], [20.0], [100.0]]], dtype=torch.bfloat16)
    output = torch.full((1, 4, 1, 2), torch.nan, dtype=torch.bfloat16)

    with patch(
        "vllm_ascend.ops.kimi_kda.get_forward_context",
        return_value=SimpleNamespace(attn_metadata={attention.prefix: metadata}),
    ):
        attention._forward(
            mixed_qkv=mixed_qkv,
            g1=torch.zeros(1, 4, 1, 2, dtype=torch.bfloat16),
            g2=torch.zeros(4, 1, 2, dtype=torch.bfloat16),
            beta=raw_beta,
            core_attn_out=output,
        )

    torch.testing.assert_close(attention._run_recurrent.call_args.args[4], raw_beta[:, :3])
    torch.testing.assert_close(output[0, :2, 0], mixed_qkv[:2, :2])
    assert torch.equal(output[:, 2:], torch.zeros_like(output[:, 2:]))


def test_overlapped_qkv_bfg_keeps_two_stage_vector_cube_overlap():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    trace: list[str] = []
    main_stream = _RecordingStream("main", ["hidden_ready", "quant_ready"], trace)
    bfg_stream = _RecordingStream("bfg", ["bfg_projection_ready", "bfg_ready"], trace)
    hidden_states = _RecordingTensor("hidden", trace)
    fused_bfg = _RecordingTensor("fused_bfg", trace)
    processed_bfg = tuple(_RecordingTensor(name, trace) for name in ("beta", "raw_gate", "output_gate"))
    quantized_qkv = object()
    qkv = object()

    def record_project_bfg(_hidden_states: object) -> _RecordingTensor:
        trace.append("project_bfg")
        return fused_bfg

    def record_dynamic_quant(_hidden_states: object) -> object:
        trace.append("dynamic_quant")
        return quantized_qkv

    def record_qkv_matmul(_qkv_input: object) -> object:
        trace.append("qkv_matmul")
        return qkv

    def record_postprocess_bfg(*_args: object) -> tuple[_RecordingTensor, ...]:
        trace.append("postprocess_bfg")
        return processed_bfg

    attention._project_bfg = MagicMock(side_effect=record_project_bfg)
    attention._quantize_fused_qkv = MagicMock(side_effect=record_dynamic_quant)
    attention._matmul_fused_qkv = MagicMock(side_effect=record_qkv_matmul)
    attention._postprocess_bfg = MagicMock(side_effect=record_postprocess_bfg)

    with (
        patch("vllm_ascend.ops.kimi_kda.torch.npu.current_stream", return_value=main_stream),
        patch("vllm_ascend.ops.kimi_kda._kda_bfg_stream", return_value=bfg_stream),
        patch(
            "vllm_ascend.ops.kimi_kda.npu_stream_switch",
            side_effect=lambda stream: _RecordingStreamSwitch(stream, trace),
        ),
    ):
        actual = attention._run_overlapped_qkv_bfg(hidden_states)

    assert actual == (qkv, *processed_bfg)
    assert trace == [
        "main.record:hidden_ready",
        "hidden.record_stream:bfg",
        "enter:bfg",
        "bfg.wait:hidden_ready",
        "project_bfg",
        "bfg.record:bfg_projection_ready",
        "exit:bfg",
        "dynamic_quant",
        "main.record:quant_ready",
        "main.wait:bfg_projection_ready",
        "qkv_matmul",
        "enter:bfg",
        "bfg.wait:quant_ready",
        "postprocess_bfg",
        "bfg.record:bfg_ready",
        "exit:bfg",
        "beta.record_stream:main",
        "raw_gate.record_stream:main",
        "output_gate.record_stream:main",
        "main.wait:bfg_ready",
    ]


@pytest.mark.parametrize(
    "quant_method_type",
    [AscendW4A8MXFPDynamicLinearMethod, AscendW8A8MXFP8DynamicLinearMethod],
)
def test_fused_qkv_splits_mxfp_dynamic_quant_from_matmul(quant_method_type):
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    inner_quant_method = quant_method_type.__new__(quant_method_type)
    if isinstance(inner_quant_method, AscendW8A8MXFP8DynamicLinearMethod):
        inner_quant_method.dynamic_mx_quant_scale_alg = "floor"
    adapter = SimpleNamespace(
        quant_method=inner_quant_method,
        apply=MagicMock(return_value=torch.randn(4, 18)),
    )
    attention.in_proj_qkvgfab = SimpleNamespace(quant_method=adapter)
    hidden_states = torch.randn(4, 6, dtype=torch.bfloat16)
    quantized = torch.empty(4, 6, dtype=torch.float8_e4m3fn)
    dynamic_scale = torch.empty(4, 1, dtype=torch.uint8)

    with patch(
        "vllm_ascend.ops.kimi_kda.torch_npu.npu_dynamic_mx_quant",
        return_value=(quantized, dynamic_scale),
    ) as dynamic_quant:
        qkv_input = attention._quantize_fused_qkv(hidden_states)

    assert isinstance(qkv_input, tuple)
    assert qkv_input[0] is quantized
    assert qkv_input[1] is dynamic_scale
    if isinstance(inner_quant_method, AscendW8A8MXFP8DynamicLinearMethod):
        dynamic_quant.assert_called_once_with(
            hidden_states,
            dst_type=torch.float8_e4m3fn,
            scale_alg="floor",
        )
    else:
        dynamic_quant.assert_called_once_with(hidden_states, dst_type=torch.float8_e4m3fn)
    output = attention._matmul_fused_qkv(qkv_input)
    assert output is adapter.apply.return_value
    adapter.apply.assert_called_once_with(attention.in_proj_qkvgfab, qkv_input, bias=None)


def test_fused_qkv_keeps_non_mxfp_quantization_in_linear_apply():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    adapter = SimpleNamespace(
        quant_method=object(),
        apply=MagicMock(return_value=torch.randn(4, 18)),
    )
    attention.in_proj_qkvgfab = SimpleNamespace(quant_method=adapter)
    hidden_states = torch.randn(4, 6)

    with patch("vllm_ascend.ops.kimi_kda.torch_npu.npu_dynamic_mx_quant") as dynamic_quant:
        qkv_input = attention._quantize_fused_qkv(hidden_states)

    assert qkv_input is hidden_states
    dynamic_quant.assert_not_called()
    output = attention._matmul_fused_qkv(qkv_input)
    assert output is adapter.apply.return_value
    adapter.apply.assert_called_once_with(attention.in_proj_qkvgfab, hidden_states, bias=None)


@pytest.mark.parametrize("compact_metadata", [False, True])
def test_prefill_pads_requests_and_restores_output_and_state(compact_metadata: bool):
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.local_num_heads = 1
    attention.head_dim = 2
    attention.gate_lower_bound = -5.0
    attention.A_log = nn.Parameter(torch.randn(1))
    attention.dt_bias = nn.Parameter(torch.randn(2))

    # A short request and one crossing a 64-token boundary share a padded
    # batch. Asymmetric state values catch accidental V/K transposition.
    q = torch.arange(134, dtype=torch.bfloat16).reshape(1, 67, 1, 2)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_gate = torch.randn_like(q)
    beta = torch.linspace(-20, 20, 67, dtype=torch.bfloat16).reshape(1, 67, 1)
    recurrent_state = torch.arange(16, dtype=torch.bfloat16).reshape(4, 1, 2, 2)
    original_state = recurrent_state.clone()
    state_indices = torch.tensor([3, 1], dtype=torch.int32)
    has_initial_state = torch.tensor([True, False])
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 2, 67),
        cu_seqlens_kern=None,
        keep_meta=None,
    )
    if compact_metadata:
        metadata.cu_seqlens_host = (0, 2, 2, 67)
        metadata.cu_seqlens_kern = torch.tensor([0, 2, 67], dtype=torch.int32)
        metadata.keep_meta = torch.tensor([True, False, True])
        state_indices = torch.tensor([3, 2, 1], dtype=torch.int32)
        has_initial_state = torch.tensor([True, True, False])
    final_state = torch.tensor([[[[1.25, 2.5], [3.75, 4.0]]], [[[5.0, 6.5], [7.25, 8.0]]]])

    def clear_initial_state(state, has_initial):
        state[~has_initial] = 0

    def flash_kda(q_batch, k_batch, v_batch, **kwargs):
        assert q_batch.shape == (2, 128, 1, 2)
        assert kwargs["initial_state"].dtype == torch.float32
        torch.testing.assert_close(kwargs["initial_state"][0], original_state[3].float())
        assert torch.count_nonzero(kwargs["initial_state"][1]) == 0
        for packed, batched in ((q, q_batch), (k, k_batch), (v, v_batch)):
            torch.testing.assert_close(batched[0, :2], packed[0, :2])
            torch.testing.assert_close(batched[1, :65], packed[0, 2:])
            assert torch.count_nonzero(batched[0, 2:]) == 0
            assert torch.count_nonzero(batched[1, 65:]) == 0
        for packed, name in ((raw_gate, "g"), (beta, "beta")):
            batched = kwargs[name]
            torch.testing.assert_close(batched[0, :2], packed[0, :2])
            torch.testing.assert_close(batched[1, :65], packed[0, 2:])
            assert torch.isneginf(batched[0, 2:]).all()
            assert torch.isneginf(batched[1, 65:]).all()
        assert kwargs["beta"].dtype == beta.dtype
        assert kwargs["lower_bound"] == -5.0
        assert kwargs["layout_qkv"] == "BSND"
        # Poison the padded output so that any failure to unpad is visible.
        output = torch.full_like(q_batch, torch.nan)
        output[0, :2] = q_batch[0, :2]
        output[1, :65] = q_batch[1, :65]
        return output, final_state

    with (
        patch("vllm_ascend.ops.kimi_kda.clear_ssm_states", side_effect=clear_initial_state),
        patch("vllm_ascend.ops.kimi_kda._flash_kda_impl", side_effect=flash_kda),
    ):
        actual = attention._run_prefill(
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state,
            state_indices,
            has_initial_state,
            metadata,
        )

    torch.testing.assert_close(actual, q)
    torch.testing.assert_close(recurrent_state[[3, 1]], final_state.to(recurrent_state.dtype))
    torch.testing.assert_close(recurrent_state[[0, 2]], original_state[[0, 2]])


@pytest.mark.parametrize("speculative", [False, True])
def test_recurrent_preserves_raw_beta_state_slots_and_token_order(speculative: bool):
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.local_num_heads = 1
    attention.head_dim = 2
    attention.gate_lower_bound = -5.0
    attention.A_log = nn.Parameter(torch.randn(1))
    attention.dt_bias = nn.Parameter(torch.randn(2))

    sequence_length = 3 if speculative else 1
    num_tokens = 2 * sequence_length
    q = torch.arange(num_tokens * 2, dtype=torch.bfloat16).reshape(1, num_tokens, 1, 2)
    k, v, gate = (torch.randn_like(q) for _ in range(3))
    beta = torch.linspace(-20, 20, num_tokens, dtype=torch.bfloat16).reshape(1, num_tokens, 1)
    state = torch.zeros(8, 1, 2, 2)
    indices = torch.tensor([[3, 4, 5], [1, 2, 6]] if speculative else [3, 1], dtype=torch.int32)
    accepted = torch.tensor([2, 1], dtype=torch.int32) if speculative else None
    query_start_loc = torch.tensor([0, sequence_length, num_tokens], dtype=torch.int32)

    def recurrent_kda(q_batch, k_batch, v_batch, state_arg, beta_batch, gate_batch, *args, **kwargs):
        assert q_batch.shape == (2, sequence_length, 1, 2)
        for packed, batched in ((q, q_batch), (k, k_batch), (v, v_batch), (gate, gate_batch)):
            torch.testing.assert_close(batched.flatten(0, 1), packed.squeeze(0))
        torch.testing.assert_close(beta_batch.reshape_as(beta), beta)
        assert state_arg is state
        torch.testing.assert_close(kwargs["ssm_state_indices"], indices.flatten())
        assert kwargs["num_accepted_tokens"] is accepted
        torch.testing.assert_close(
            kwargs["query_lengths"],
            torch.tensor([sequence_length, sequence_length], dtype=torch.int32),
        )
        return q_batch.clone()

    with patch("vllm_ascend.ops.kimi_kda._recurrent_kda_impl", side_effect=recurrent_kda):
        actual = attention._run_recurrent(
            q,
            k,
            v,
            gate,
            beta,
            state,
            query_start_loc,
            indices,
            num_accepted_tokens=accepted,
        )

    torch.testing.assert_close(actual, q)


@pytest.mark.parametrize("num_input_tokens", [3, 4])
def test_recurrent_unpacks_short_drafts_and_skips_empty_rows(num_input_tokens: int):
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.local_num_heads = 1
    attention.head_dim = 2
    attention.gate_lower_bound = -5.0
    attention.A_log = nn.Parameter(torch.randn(1))
    attention.dt_bias = nn.Parameter(torch.randn(2))

    q = torch.arange(num_input_tokens * 2, dtype=torch.bfloat16).reshape(1, num_input_tokens, 1, 2)
    k, v, gate = (torch.randn_like(q) for _ in range(3))
    beta = torch.linspace(-20, 20, num_input_tokens, dtype=torch.bfloat16).reshape(1, num_input_tokens, 1)
    # MRV2 pages can contain other cache data between recurrent states.
    state_pages = torch.arange(64, dtype=torch.float32).reshape(8, 2, 1, 2, 2)
    state = state_pages[:, 0]
    assert not state.is_contiguous()
    indices = torch.tensor([[3, 4, -1], [1, -1, -1], [-1, -1, -1]], dtype=torch.int32)
    accepted = torch.tensor([1, 1, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 3, 3], dtype=torch.int32)

    def recurrent_kda(q_batch, k_batch, v_batch, state_arg, beta_batch, gate_batch, *args, **kwargs):
        assert q_batch.shape == (3, 3, 1, 2)
        assert state_arg is state
        torch.testing.assert_close(kwargs["ssm_state_indices"], indices.flatten())
        torch.testing.assert_close(kwargs["query_lengths"], torch.tensor([2, 1, 0], dtype=torch.int32))
        assert kwargs["num_accepted_tokens"] is accepted
        valid = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
        for packed, batched in ((q, q_batch), (k, k_batch), (v, v_batch)):
            torch.testing.assert_close(batched[0, :2], packed[0, :2])
            torch.testing.assert_close(batched[1, :1], packed[0, 2:3])
            assert torch.count_nonzero(batched[~valid]) == 0
        torch.testing.assert_close(beta_batch[0, :2, :, 0], beta[0, :2])
        torch.testing.assert_close(beta_batch[1, :1, :, 0], beta[0, 2:3])
        assert torch.isneginf(beta_batch[~valid]).all()
        torch.testing.assert_close(gate_batch[0, :2], gate[0, :2])
        torch.testing.assert_close(gate_batch[1, :1], gate[0, 2:3])
        assert torch.isneginf(gate_batch[~valid]).all()
        # Inactive kernel rows are undefined; neither request padding nor
        # an empty sequence may leak them into the packed output.
        output = torch.full_like(q_batch, torch.nan)
        output[valid] = q_batch[valid] + 10
        return output

    with patch("vllm_ascend.ops.kimi_kda._recurrent_kda_impl", side_effect=recurrent_kda):
        actual = attention._run_recurrent(
            q,
            k,
            v,
            gate,
            beta,
            state,
            query_start_loc,
            indices,
            num_accepted_tokens=accepted,
        )

    assert actual.shape == q.shape
    torch.testing.assert_close(actual[:, :3], q[:, :3] + 10)
    assert torch.equal(actual[:, 3:], torch.zeros_like(actual[:, 3:]))
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("run_mode", [0, 1])
def test_causal_conv_dispatches_initial_state_and_accepted_tokens(run_mode: int):
    mixed_qkv = torch.randn(4, 6)
    weight = torch.randn(4, 6)
    state = torch.randn(8, 6, 3)
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    indices = torch.tensor([[3, 4], [1, 2]], dtype=torch.int32)
    initial_state = torch.tensor([True, False])
    accepted = torch.tensor([2, 1], dtype=torch.int32)
    expected = torch.randn_like(mixed_qkv)
    op_name = "causal_conv1d_fn" if run_mode == 0 else "causal_conv1d_update"

    with patch.object(torch.ops.cann_ops_transformer, op_name, return_value=expected, create=True) as conv:
        actual = AscendKimiK3DeltaAttention._run_causal_conv1d(
            mixed_qkv,
            weight,
            state,
            query_start_loc,
            indices,
            initial_state if run_mode == 0 else None,
            run_mode=run_mode,
            num_accepted_tokens=accepted if run_mode == 1 else None,
        )

    assert actual is expected
    kwargs = conv.call_args.kwargs
    assert kwargs["x"] is mixed_qkv
    assert kwargs["weight"] is weight
    assert kwargs["query_start_loc"] is query_start_loc
    if run_mode == 0:
        assert kwargs["conv_states"] is state
        torch.testing.assert_close(kwargs["cache_indices"], indices[:, 0])
        torch.testing.assert_close(kwargs["has_initial_state"], initial_state.to(torch.int32))
    else:
        assert kwargs["conv_state"] is state
        torch.testing.assert_close(kwargs["conv_state_indices"], indices[:, 0])
        assert kwargs["num_accepted_tokens"] is accepted


@pytest.mark.parametrize("run_mode", [0, 1])
def test_causal_conv_updates_only_selected_pages_in_strided_state(run_mode: int):
    mixed_qkv = torch.randn(4, 6)
    weight = torch.randn(4, 6)
    state_pages = torch.arange(216, dtype=torch.float32).reshape(6, 2, 6, 3)
    state = state_pages[:, 0]
    original_pages = state_pages.clone()
    assert not state.is_contiguous()
    query_start_loc = torch.tensor([0, 2, 4, 4], dtype=torch.int32)
    indices = torch.tensor([3, 1, 0], dtype=torch.int32)
    initial_state = torch.tensor([True, True, False])
    accepted = torch.tensor([2, 1, 0], dtype=torch.int32)
    expected_output = torch.randn_like(mixed_qkv)
    op_name = "causal_conv1d_fn" if run_mode == 0 else "causal_conv1d_update"

    def causal_conv(**kwargs):
        kernel_state = kwargs["conv_states" if run_mode == 0 else "conv_state"]
        kernel_indices = kwargs["cache_indices" if run_mode == 0 else "conv_state_indices"]
        assert kernel_state.is_contiguous()
        assert kernel_state.shape == (4, 6, 3)
        torch.testing.assert_close(kernel_indices, torch.tensor([1, 2, 0], dtype=torch.int32))
        assert torch.count_nonzero(kernel_state[0]) == 0
        torch.testing.assert_close(kernel_state[1], original_pages[3, 0])
        torch.testing.assert_close(kernel_state[2], original_pages[1, 0])
        kernel_state[1].add_(100)
        kernel_state[2].add_(200)
        return expected_output

    with patch.object(torch.ops.cann_ops_transformer, op_name, side_effect=causal_conv, create=True):
        actual = AscendKimiK3DeltaAttention._run_causal_conv1d(
            mixed_qkv,
            weight,
            state,
            query_start_loc,
            indices,
            initial_state if run_mode == 0 else None,
            run_mode=run_mode,
            num_accepted_tokens=accepted if run_mode == 1 else None,
        )

    assert actual is expected_output
    expected_pages = original_pages.clone()
    expected_pages[3, 0].add_(100)
    expected_pages[1, 0].add_(200)
    # Comparing the backing pages also catches changes to the null slot,
    # unselected requests, or the data between consecutive cache states.
    torch.testing.assert_close(state_pages, expected_pages)
    torch.testing.assert_close(state[0], original_pages[0, 0])


@pytest.mark.parametrize("run_mode", [0, 1])
@pytest.mark.parametrize("strided", [False, True])
def test_causal_conv_maps_empty_queries_to_null_slot(run_mode: int, strided: bool):
    mixed_qkv = torch.randn(1, 6)
    state_pages = torch.arange(144, dtype=torch.float32).reshape(4, 2, 6, 3)
    state = state_pages[:, 0]
    if not strided:
        state = state.contiguous()
    original_state = state.clone()
    indices = torch.tensor([2, 3], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 1], dtype=torch.int32)
    op_name = "causal_conv1d_fn" if run_mode == 0 else "causal_conv1d_update"
    expected_output = torch.randn_like(mixed_qkv)

    def causal_conv(**kwargs):
        kernel_state = kwargs["conv_states" if run_mode == 0 else "conv_state"]
        kernel_indices = kwargs["cache_indices" if run_mode == 0 else "conv_state_indices"]
        expected_indices = torch.tensor([1 if strided else 2, 0], dtype=torch.int32)
        torch.testing.assert_close(kernel_indices, expected_indices)
        # CANN can touch every non-null state even when its query is empty.
        kernel_state.index_fill_(0, kernel_indices[kernel_indices != 0].long(), -123)
        return expected_output

    with patch.object(torch.ops.cann_ops_transformer, op_name, side_effect=causal_conv, create=True):
        actual = AscendKimiK3DeltaAttention._run_causal_conv1d(
            mixed_qkv,
            torch.randn(4, 6),
            state,
            query_start_loc,
            indices,
            torch.tensor([True, True]) if run_mode == 0 else None,
            run_mode=run_mode,
            num_accepted_tokens=torch.tensor([1, 0], dtype=torch.int32) if run_mode == 1 else None,
        )

    assert actual is expected_output
    expected_state = original_state.clone()
    expected_state[2].fill_(-123)
    torch.testing.assert_close(state, expected_state)
    torch.testing.assert_close(state[3], original_state[3])


def test_kda_empty_forward_context_clears_preallocated_output():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    core_attn_out = torch.full((1, 4, 2, 3), torch.nan)

    with patch(
        "vllm_ascend.ops.kimi_kda.get_forward_context",
        return_value=SimpleNamespace(attn_metadata=None),
    ):
        attention._forward(
            mixed_qkv=torch.empty(4, 18),
            g1=torch.empty(1, 4, 2, 3),
            g2=torch.empty(4, 2, 3),
            beta=torch.empty(1, 4, 2),
            core_attn_out=core_attn_out,
        )

    assert torch.equal(core_attn_out, torch.zeros_like(core_attn_out))


def test_kda_conv_weight_is_packed_once_in_kernel_layout():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.conv_size = 4
    attention.local_projection_size = 6
    attention.conv1d = nn.Module()
    source = torch.arange(18 * 4, dtype=torch.float32).reshape(18, 1, 4)
    attention.conv1d.weight = nn.Parameter(source)
    attention.register_parameter(
        _PACKED_CONV_WEIGHT_NAME,
        nn.Parameter(torch.empty(4, 18, dtype=torch.bfloat16), requires_grad=False),
    )
    original = attention.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    original_ptr = original.data_ptr()

    attention._pack_conv_weights()

    packed = attention.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    assert packed.data_ptr() == original_ptr
    assert packed.dtype == torch.bfloat16
    assert packed.is_contiguous()
    torch.testing.assert_close(
        packed,
        source[:, 0, :].transpose(0, 1).to(torch.bfloat16),
    )
