#!/usr/bin/env bash
set -euo pipefail

minimum_free_mib="${BRT_MIN_FREE_MIB:-2048}"
compute_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
gpu_row="$(nvidia-smi --query-gpu=memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | head -n 1)"
IFS=',' read -r free_mib utilization temperature <<<"${gpu_row}"
free_mib="${free_mib// /}"
utilization="${utilization// /}"
temperature="${temperature// /}"

if [[ -n "${compute_apps}" ]]; then
  echo "BRT GPU preflight refused: active compute applications detected" >&2
  echo "${compute_apps}" >&2
  exit 20
fi
if (( free_mib < minimum_free_mib )); then
  echo "BRT GPU preflight refused: free=${free_mib}MiB required=${minimum_free_mib}MiB" >&2
  exit 21
fi
if (( utilization > 5 )); then
  echo "BRT GPU preflight refused: utilization=${utilization}%" >&2
  exit 22
fi

printf 'gpu_preflight=ok free_mib=%s utilization=%s temperature=%s\n' \
  "${free_mib}" "${utilization}" "${temperature}"
