#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${BRT_TARGET:-charles@192.168.124.8}"
destination="${BRT_TARGET_DIR:-/home/charles/brt-workspace}"

if ! [[ "${target}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*$ ]]; then
  echo "BRT target refused: invalid target format" >&2
  exit 40
fi
if ! [[ "${destination}" =~ ^/home/charles/brt-[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] \
  || [[ "${destination}" == *'//' || "${destination}" == *'/./'* || "${destination}" == *'/../'* \
    || "${destination}" == */. || "${destination}" == */.. ]]; then
  echo "BRT target refused: invalid project destination" >&2
  exit 41
fi

printf -v remote_destination '%q' "${destination}"
ssh "${target}" "mkdir -p -- ${remote_destination}"
rsync -a \
  --exclude .git --exclude build --exclude target --exclude evidence/local \
  "${repo_root}/" "${target}:${destination}/"
printf 'synced_target=%s synced_dir=%s\n' "${target}" "${destination}"
