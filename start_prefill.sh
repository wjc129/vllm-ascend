#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_HOME="${VLLM_ASCEND_HOME:-${SCRIPT_DIR}}"
MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv4}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-beijing}"
NIC_NAME="${NIC_NAME:-}"
PREFILL_NODE_IP="${PREFILL_NODE_IP:-}"
PREFILL_START_PORT="${PREFILL_START_PORT:-7100}"
PREFILL_DP_RPC_PORT="${PREFILL_DP_RPC_PORT:-12321}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-36000}"
PREFILL_ENGINE_ID="${PREFILL_ENGINE_ID:-0}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/prefill}"

PREFILL_DP_SIZE=8
PREFILL_TP_SIZE=2
DECODE_DP_SIZE=16
DECODE_TP_SIZE=1

detect_local_ip() {
    local detected_ip
    detected_ip="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}')"
    if [[ -z "${detected_ip}" ]]; then
        detected_ip="$(hostname -I | awk '{print $1}')"
    fi
    printf '%s' "${detected_ip}"
}

detect_nic() {
    ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}'
}

activate_runtime() {
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV_NAME}"
    fi
    if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
        # shellcheck disable=SC1091
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]]; then
        # shellcheck disable=SC1091
        source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
    fi
}

activate_runtime
PREFILL_NODE_IP="${PREFILL_NODE_IP:-$(detect_local_ip)}"
NIC_NAME="${NIC_NAME:-$(detect_nic)}"
: "${PREFILL_NODE_IP:?Set PREFILL_NODE_IP to the prefill service IP}"
: "${NIC_NAME:?Set NIC_NAME to the HCCL/GLOO service interface}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1},${PREFILL_NODE_IP}"
export no_proxy="${NO_PROXY}"

export PYTHONPATH="${VLLM_ASCEND_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export HCCL_IF_IP="${PREFILL_NODE_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=2560
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_OP_INIT_MODE=1

# Enable LoPT and print the disabled/enabled timing comparison for long prompts.
export VLLM_ASCEND_LOPT_ENABLE="${VLLM_ASCEND_LOPT_ENABLE:-1}"
export VLLM_ASCEND_LOPT_VERIFY="${VLLM_ASCEND_LOPT_VERIFY:-1}"
export VLLM_ASCEND_LOPT_THREAD_WORKERS="${VLLM_ASCEND_LOPT_THREAD_WORKERS:-4}"
export VLLM_ASCEND_LOPT_MIN_CHARS="${VLLM_ASCEND_LOPT_MIN_CHARS:-32768}"

if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

KV_TRANSFER_CONFIG="$(cat <<JSON
{
  "kv_connector": "MooncakeHybridConnector",
  "kv_role": "kv_producer",
  "kv_port": "${PREFILL_KV_PORT}",
  "engine_id": "${PREFILL_ENGINE_ID}",
  "kv_connector_extra_config": {
    "prefill": {"dp_size": ${PREFILL_DP_SIZE}, "tp_size": ${PREFILL_TP_SIZE}},
    "decode": {"dp_size": ${DECODE_DP_SIZE}, "tp_size": ${DECODE_TP_SIZE}}
  }
}
JSON
)"

mkdir -p "${LOG_DIR}"
pids=()

cleanup() {
    trap - EXIT INT TERM
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

for ((rank = 0; rank < PREFILL_DP_SIZE; rank++)); do
    first_device=$((rank * PREFILL_TP_SIZE))
    visible_devices="${first_device},$((first_device + 1))"
    server_port=$((PREFILL_START_PORT + rank))
    log_file="${LOG_DIR}/prefill_dp${rank}.log"

    echo "Starting prefill DP rank ${rank}: devices=${visible_devices} port=${server_port} log=${log_file}"
    (
        export ASCEND_RT_VISIBLE_DEVICES="${visible_devices}"
        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port "${server_port}" \
            --data-parallel-size "${PREFILL_DP_SIZE}" \
            --data-parallel-rank "${rank}" \
            --data-parallel-address "${PREFILL_NODE_IP}" \
            --data-parallel-rpc-port "${PREFILL_DP_RPC_PORT}" \
            --tensor-parallel-size "${PREFILL_TP_SIZE}" \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name "${SERVED_MODEL_NAME}" \
            --max-model-len 1048576 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 16 \
            --no-disable-hybrid-kv-cache-manager \
            --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 128}' \
            --no-enable-prefix-caching \
            --safetensors-load-strategy prefetch \
            --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
            --trust-remote-code \
            --block-size 128 \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --enforce-eager \
            --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true, "enable_dsa_cp": true}' \
            --kv-transfer-config "${KV_TRANSFER_CONFIG}"
    ) >"${log_file}" 2>&1 &
    pids+=("$!")
done

echo "Prefill endpoints: ${PREFILL_NODE_IP}:${PREFILL_START_PORT}-$((PREFILL_START_PORT + PREFILL_DP_SIZE - 1))"
echo "LoPT timing output is written to ${LOG_DIR}/prefill_dp*.log"
set +e
wait -n "${pids[@]}"
status=$?
set -e
echo "A prefill rank exited with status ${status}; stopping the remaining ranks." >&2
exit "${status}"
