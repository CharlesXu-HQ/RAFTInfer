#!/usr/bin/env bash
set -euo pipefail

safety_refusal() {
  echo "BRT GPU preflight refused: $1" >&2
  exit 23
}

is_nonnegative_decimal() {
  [[ "$1" =~ ^[0-9]+$ && ${#1} -le 9 ]]
}

minimum_free_mib="${BRT_MIN_FREE_MIB:-2048}"
maximum_utilization="${BRT_MAX_UTILIZATION_PERCENT:-5}"
is_nonnegative_decimal "${minimum_free_mib}" || safety_refusal "invalid BRT_MIN_FREE_MIB"
is_nonnegative_decimal "${maximum_utilization}" || safety_refusal "invalid BRT_MAX_UTILIZATION_PERCENT"
(( 10#${minimum_free_mib} > 0 )) || safety_refusal "invalid BRT_MIN_FREE_MIB"
(( 10#${maximum_utilization} <= 100 )) || safety_refusal "invalid BRT_MAX_UTILIZATION_PERCENT"

if ! compute_apps="$(nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)"; then
  safety_refusal "unable to query compute applications on GPU 0"
fi
if ! gpu_row="$(nvidia-smi --id=0 --query-gpu=memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
  safety_refusal "unable to query GPU 0"
fi

[[ -n "${gpu_row}" && "${gpu_row}" != *$'\n'* ]] || safety_refusal "malformed GPU 0 query result"
comma_count="${gpu_row//[^,]/}"
[[ "${#comma_count}" -eq 2 ]] || safety_refusal "malformed GPU 0 query result"
IFS=',' read -r -a gpu_fields <<<"${gpu_row}"
[[ "${#gpu_fields[@]}" -eq 3 ]] || safety_refusal "malformed GPU 0 query result"

free_mib="${gpu_fields[0]//[[:space:]]/}"
utilization="${gpu_fields[1]//[[:space:]]/}"
temperature="${gpu_fields[2]//[[:space:]]/}"
is_nonnegative_decimal "${free_mib}" || safety_refusal "malformed GPU 0 query result"
is_nonnegative_decimal "${utilization}" || safety_refusal "malformed GPU 0 query result"
is_nonnegative_decimal "${temperature}" || safety_refusal "malformed GPU 0 query result"

if [[ -n "${compute_apps}" ]]; then
  echo "BRT GPU preflight refused: active compute applications detected" >&2
  echo "${compute_apps}" >&2
  exit 20
fi
if (( 10#${free_mib} < 10#${minimum_free_mib} )); then
  echo "BRT GPU preflight refused: free=${free_mib}MiB required=${minimum_free_mib}MiB" >&2
  exit 21
fi
if (( 10#${utilization} > 10#${maximum_utilization} )); then
  echo "BRT GPU preflight refused: utilization=${utilization}%" >&2
  exit 22
fi

printf 'gpu_preflight=ok free_mib=%s utilization=%s temperature=%s\n' \
  "${free_mib}" "${utilization}" "${temperature}"
