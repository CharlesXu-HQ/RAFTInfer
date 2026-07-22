#!/usr/bin/env bash
set -euo pipefail
target="${BRT_TARGET:-charles@192.168.124.8}"
destination="${BRT_TARGET_DIR:-/home/charles/brt-workspace}"
ssh "${target}" "mkdir -p '${destination}'"
rsync -a \
  --exclude .git --exclude build --exclude target --exclude evidence/local \
  ./ "${target}:${destination}/"
printf 'synced_target=%s synced_dir=%s\n' "${target}" "${destination}"
