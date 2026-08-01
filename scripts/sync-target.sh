#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${RAFTINFER_TARGET:-}"
destination="${RAFTINFER_TARGET_DIR:-}"

if [[ -z "${target}" ]] || ! [[ "${target}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*$ ]]; then
  echo "RAFTINFER target refused: a valid RAFTINFER_TARGET is required" >&2
  exit 40
fi
if [[ -z "${destination}" ]] \
  || ! [[ "${destination}" =~ ^/[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$ ]]; then
  echo "RAFTINFER target refused: a safe RAFTINFER_TARGET_DIR is required" >&2
  exit 41
fi

printf -v remote_destination '%q' "${destination}"
ssh "${target}" "mkdir -p -- ${remote_destination}"
rsync -a \
  --exclude .git --exclude build --exclude target --exclude evidence/local \
  "${repo_root}/" "${target}:${destination}/"
printf 'synced_target=%s synced_dir=%s\n' "${target}" "${destination}"
