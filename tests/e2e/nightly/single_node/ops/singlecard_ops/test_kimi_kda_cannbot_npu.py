# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Atlas 950 coverage for CANNBot KDA query lengths and paged state views."""

import pytest
import torch

_STATE_CAPACITY = 16


@pytest.fixture(scope="module")
def cannbot_recurrent():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    if "950" not in torch.npu.get_device_name(torch.npu.current_device()):
        pytest.skip("CANNBot KDA requires Atlas 950")
    pytest.importorskip("cannbotdsl")

    from ops.cannbot_dsl.fused_recurrent_kda import fused_recurrent_kda_op

    return fused_recurrent_kda_op


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16], ids=["fp32_state", "bf16_state"])
@pytest.mark.parametrize("invalid_snapshot", [-1, _STATE_CAPACITY], ids=["negative_slot", "past_pool_slot"])
@torch.inference_mode()
def test_recurrent_kda_short_queries_preserve_paged_cache(cannbot_recurrent, state_dtype, invalid_snapshot):
    torch.manual_seed(20260914)
    device = torch.device("npu")
    batch, width, heads, dim = 3, 4, 4, 128
    lengths = [2, 1, 0]
    accepted = [4, 3, 0]
    state_indices = [
        [2, invalid_snapshot, 8, 9],
        [5, 6, 10, 11],
        [-1, -1, -1, -1],
    ]

    shape = (batch, width, heads, dim)
    q_cpu = torch.randn(shape, dtype=torch.bfloat16)
    k_cpu = torch.randn(shape, dtype=torch.bfloat16)
    v_cpu = torch.randn(shape, dtype=torch.bfloat16)
    gate_cpu = torch.randn(shape, dtype=torch.bfloat16) * 0.25
    beta_cpu = torch.randn(batch, width, heads, 1, dtype=torch.bfloat16)
    for tensor in (q_cpu, k_cpu, v_cpu, gate_cpu, beta_cpu):
        for request, length in enumerate(lengths):
            tensor[request, length:] = torch.nan
    q, k, v, gate, beta = (tensor.to(device) for tensor in (q_cpu, k_cpu, v_cpu, gate_cpu, beta_cpu))
    a_log = torch.zeros(heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(heads, dim, dtype=torch.float32, device=device)

    # Distinct initial values make choosing column zero instead of the
    # accepted snapshot observable even when the current query is shorter.
    initial_cpu = torch.randn(_STATE_CAPACITY, heads, dim, dim) * 0.05
    initial_cpu += torch.arange(_STATE_CAPACITY).view(-1, 1, 1, 1) * 0.2
    initial_cpu = initial_cpu.to(state_dtype)
    backing = torch.full(
        (_STATE_CAPACITY + 2, 2, heads, dim, dim),
        7.0,
        dtype=state_dtype,
        device=device,
    )
    state = backing[1 : _STATE_CAPACITY + 1, 0]
    state.copy_(initial_cpu.to(device))
    backing_before = backing.cpu().clone()
    original_stride = state.stride()
    original_storage = state.untyped_storage().data_ptr()
    assert not state.is_contiguous()
    assert state.storage_offset() > 0

    expected_output = torch.zeros_like(v_cpu)
    expected_state = initial_cpu.clone()
    written_slots = set()
    for request, length in enumerate(lengths):
        if length == 0:
            continue
        initial_slot = state_indices[request][accepted[request] - 1]
        # Run the dense API without query_lengths or accepted-token metadata.
        # Seed its first slot from the independently selected previous-step
        # snapshot, then map each dense snapshot back to its valid global slot.
        reference_state = torch.zeros(length, heads, dim, dim, dtype=state_dtype, device=device)
        reference_state[0].copy_(initial_cpu[initial_slot].to(device))
        reference_output = cannbot_recurrent(
            q[request : request + 1, :length].contiguous(),
            k[request : request + 1, :length].contiguous(),
            v[request : request + 1, :length].contiguous(),
            reference_state,
            beta[request : request + 1, :length].contiguous(),
            gate[request : request + 1, :length].contiguous(),
            dim**-0.5,
            a_log,
            dt_bias,
            -1.0,
            "BSND",
        )
        torch.npu.synchronize()
        expected_output[request, :length] = reference_output.cpu()[0]
        reference_state_cpu = reference_state.cpu()
        for token in range(length):
            slot = state_indices[request][token]
            if 0 <= slot < _STATE_CAPACITY:
                expected_state[slot] = reference_state_cpu[token]
                written_slots.add(slot)

    output = cannbot_recurrent(
        q,
        k,
        v,
        state,
        beta,
        gate,
        dim**-0.5,
        a_log,
        dt_bias,
        -1.0,
        "BSND",
        ssm_state_indices=torch.tensor(state_indices, dtype=torch.int32, device=device).reshape(-1),
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32, device=device),
        query_lengths=torch.tensor(lengths, dtype=torch.int32, device=device),
    )
    torch.npu.synchronize()
    output_cpu = output.cpu()
    state_cpu = state.cpu()
    backing_after = backing.cpu()

    torch.testing.assert_close(output_cpu, expected_output, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(state_cpu, expected_state, rtol=2e-2, atol=2e-3)
    assert torch.isfinite(output_cpu).all()
    for request, length in enumerate(lengths):
        torch.testing.assert_close(
            output_cpu[request, length:],
            torch.zeros_like(output_cpu[request, length:]),
            rtol=0,
            atol=0,
        )

    # Check both neighboring guard pages, the interleaved page padding, and
    # every unselected state slot exactly, including slots in the empty row.
    assert state.stride() == original_stride
    assert state.untyped_storage().data_ptr() == original_storage
    torch.testing.assert_close(backing_after[0], backing_before[0], rtol=0, atol=0)
    torch.testing.assert_close(backing_after[-1], backing_before[-1], rtol=0, atol=0)
    torch.testing.assert_close(backing_after[:, 1], backing_before[:, 1], rtol=0, atol=0)
    untouched_slots = [slot for slot in range(_STATE_CAPACITY) if slot not in written_slots]
    torch.testing.assert_close(state_cpu[untouched_slots], initial_cpu[untouched_slots], rtol=0, atol=0)
