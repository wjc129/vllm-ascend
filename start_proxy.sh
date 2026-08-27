#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_HOME="${VLLM_ASCEND_HOME:-${SCRIPT_DIR}}"
PREFILL_NODE_IP="${PREFILL_NODE_IP:-7.150.2.170}"
DECODE_NODE_IP="${DECODE_NODE_IP:-7.150.1.10}"
PREFILL_START_PORT="${PREFILL_START_PORT:-7100}"
DECODE_START_PORT="${DECODE_START_PORT:-7100}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
PROXY_PORT="${PROXY_PORT:-8000}"
PROXY_WORKERS="${PROXY_WORKERS:-1}"
PROXY_SCRIPT="${PROXY_SCRIPT:-${VLLM_ASCEND_HOME}/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py}"
PROXY_LOG_FILE="${PROXY_LOG_FILE:-${SCRIPT_DIR}/logs/proxy/proxy.log}"
PROXY_PID_FILE="${PROXY_PID_FILE:-${SCRIPT_DIR}/run/proxy/proxy.pid}"

PREFILL_DP_SIZE=8
DECODE_DP_SIZE=16

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1},${PREFILL_NODE_IP},${DECODE_NODE_IP}"
export no_proxy="${NO_PROXY}"
export PYTHONPATH="${VLLM_ASCEND_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

if [[ ! -f "${PROXY_SCRIPT}" ]]; then
    echo "Proxy script not found: ${PROXY_SCRIPT}" >&2
    exit 1
fi
if [[ "${PREFILL_NODE_IP}" == "prefill-node" ]] && ! getent hosts "${PREFILL_NODE_IP}" >/dev/null 2>&1; then
    echo "Set PREFILL_NODE_IP to the real prefill node IP, or add prefill-node to /etc/hosts." >&2
    exit 1
fi
if [[ "${DECODE_NODE_IP}" == "decode-node" ]] && ! getent hosts "${DECODE_NODE_IP}" >/dev/null 2>&1; then
    echo "Set DECODE_NODE_IP to the real decode node IP, or add decode-node to /etc/hosts." >&2
    exit 1
fi

prefiller_hosts=()
prefiller_ports=()
for ((rank = 0; rank < PREFILL_DP_SIZE; rank++)); do
    prefiller_hosts+=("${PREFILL_NODE_IP}")
    prefiller_ports+=("$((PREFILL_START_PORT + rank))")
done

decoder_hosts=()
decoder_ports=()
for ((rank = 0; rank < DECODE_DP_SIZE; rank++)); do
    decoder_hosts+=("${DECODE_NODE_IP}")
    decoder_ports+=("$((DECODE_START_PORT + rank))")
done

echo "Starting PD proxy on ${PROXY_HOST}:${PROXY_PORT}"
echo "Prefill: ${PREFILL_NODE_IP}:${PREFILL_START_PORT}-$((PREFILL_START_PORT + PREFILL_DP_SIZE - 1))"
echo "Decode:  ${DECODE_NODE_IP}:${DECODE_START_PORT}-$((DECODE_START_PORT + DECODE_DP_SIZE - 1))"

mkdir -p "$(dirname -- "${PROXY_LOG_FILE}")" "$(dirname -- "${PROXY_PID_FILE}")"
if [[ -f "${PROXY_PID_FILE}" ]]; then
    existing_pid="$(<"${PROXY_PID_FILE}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "Proxy is already running: pid=${existing_pid} log=${PROXY_LOG_FILE}"
        exit 0
    fi
    rm -f "${PROXY_PID_FILE}"
fi

nohup python3 "${PROXY_SCRIPT}" \
    --host "${PROXY_HOST}" \
    --port "${PROXY_PORT}" \
    --prefiller-hosts "${prefiller_hosts[@]}" \
    --prefiller-ports "${prefiller_ports[@]}" \
    --decoder-hosts "${decoder_hosts[@]}" \
    --decoder-ports "${decoder_ports[@]}" \
    --workers "${PROXY_WORKERS}" \
    --log-level info \
    >"${PROXY_LOG_FILE}" 2>&1 </dev/null &
proxy_pid=$!
printf '%s\n' "${proxy_pid}" >"${PROXY_PID_FILE}"
echo "Proxy started: pid=${proxy_pid} log=${PROXY_LOG_FILE}"
echo "Follow logs with: tail -F ${PROXY_LOG_FILE}"
