#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_HOME="${VLLM_ASCEND_HOME:-${SCRIPT_DIR}}"
MODEL_PATH="${MODEL_PATH:-/data/models/deepseekmtp}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv4}"
NIC_NAME="${NIC_NAME:-enp23s0f3}"
DECODE_NODE_IP="${DECODE_NODE_IP:-7.150.1.10}"
DECODE_START_PORT="${DECODE_START_PORT:-7100}"
DECODE_DP_RPC_PORT="${DECODE_DP_RPC_PORT:-12321}"
DECODE_KV_PORT="${DECODE_KV_PORT:-36100}"
DECODE_ENGINE_ID="${DECODE_ENGINE_ID:-1}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/decode}"
PID_DIR="${PID_DIR:-${SCRIPT_DIR}/run/decode}"

PREFILL_DP_SIZE=8
PREFILL_TP_SIZE=2
DECODE_DP_SIZE=16
DECODE_TP_SIZE=1

activate_runtime() {
    if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
        # shellcheck disable=SC1091
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]]; then
        # shellcheck disable=SC1091
        source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
    fi
}

activate_runtime
: "${DECODE_NODE_IP:?Set DECODE_NODE_IP to the decode service IP}"
: "${NIC_NAME:?Set NIC_NAME to the HCCL/GLOO service interface}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1},${DECODE_NODE_IP}"
export no_proxy="${NO_PROXY}"

export PYTHONPATH="${VLLM_ASCEND_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export HCCL_IF_IP="${DECODE_NODE_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_OP_INIT_MODE=1

# Decode receives token IDs from the prefiller; LoPT timing is collected on P only.
export VLLM_ASCEND_LOPT_ENABLE=0
export VLLM_ASCEND_LOPT_VERIFY=0

if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

KV_TRANSFER_CONFIG="$(cat <<JSON
{
  "kv_connector": "MooncakeHybridConnector",
  "kv_role": "kv_consumer",
  "kv_port": "${DECODE_KV_PORT}",
  "engine_id": "${DECODE_ENGINE_ID}",
  "kv_connector_extra_config": {
    "prefill": {"dp_size": ${PREFILL_DP_SIZE}, "tp_size": ${PREFILL_TP_SIZE}},
    "decode": {"dp_size": ${DECODE_DP_SIZE}, "tp_size": ${DECODE_TP_SIZE}}
  }
}
JSON
)"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

for ((rank = 0; rank < DECODE_DP_SIZE; rank++)); do
    visible_devices="${rank}"
    server_port=$((DECODE_START_PORT + rank))
    log_file="${LOG_DIR}/decode_dp${rank}.log"
    pid_file="${PID_DIR}/decode_dp${rank}.pid"

    if [[ -f "${pid_file}" ]]; then
        existing_pid="$(<"${pid_file}")"
        if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
            echo "Decode DP rank ${rank} is already running: pid=${existing_pid}"
            continue
        fi
        rm -f "${pid_file}"
    fi

    echo "Starting decode DP rank ${rank}: device=${visible_devices} port=${server_port} log=${log_file}"
    nohup env ASCEND_RT_VISIBLE_DEVICES="${visible_devices}" \
        vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port "${server_port}" \
            --data-parallel-size "${DECODE_DP_SIZE}" \
            --data-parallel-rank "${rank}" \
            --data-parallel-address "${DECODE_NODE_IP}" \
            --data-parallel-rpc-port "${DECODE_DP_RPC_PORT}" \
            --tensor-parallel-size "${DECODE_TP_SIZE}" \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name "${SERVED_MODEL_NAME}" \
            --max-model-len 1048576 \
            --max-num-batched-tokens 120 \
            --max-num-seqs 60 \
            --async-scheduling \
            --block-size 128 \
            --no-disable-hybrid-kv-cache-manager \
            --no-enable-prefix-caching \
            --safetensors-load-strategy prefetch \
            --trust-remote-code \
            --tokenizer-mode deepseek_v4 \
            --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 128}' \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
            --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "enable_cpu_binding": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true}' \
        >"${log_file}" 2>&1 </dev/null &
    process_pid=$!
    printf '%s\n' "${process_pid}" >"${pid_file}"
    echo "Decode DP rank ${rank} started: pid=${process_pid}"
done

echo "Decode endpoints: ${DECODE_NODE_IP}:${DECODE_START_PORT}-$((DECODE_START_PORT + DECODE_DP_SIZE - 1))"
echo "Follow logs with: tail -F ${LOG_DIR}/decode_dp*.log"
