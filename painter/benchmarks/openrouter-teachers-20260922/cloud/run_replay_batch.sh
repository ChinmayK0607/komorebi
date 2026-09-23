#!/usr/bin/env bash
# Replay saved timeout programs across public shards with one cloud setup.
set -Eeuo pipefail

if [[ $# -ne 4 ]]; then
  echo 'Usage: run_replay_batch.sh TRACK SHARDS_CSV TIMEOUT_SECONDS WORKERS' >&2
  echo 'Example: run_replay_batch.sh speed 00,01,02,03,04 360 2' >&2
  exit 2
fi
track="$1"
csv="$2"
timeout="$3"
workers="$4"
[[ "$track" == quality || "$track" == speed ]] || { echo 'Invalid track' >&2; exit 2; }
[[ "$timeout" =~ ^[0-9]+$ && "$timeout" -ge 181 && "$timeout" -le 900 ]] || { echo 'Invalid timeout' >&2; exit 2; }
[[ "$workers" =~ ^[0-9]+$ && "$workers" -ge 1 && "$workers" -le 8 ]] || { echo 'Invalid worker count' >&2; exit 2; }
IFS=, read -r -a shards <<< "$csv"
[[ ${#shards[@]} -ge 1 ]] || { echo 'No shards selected' >&2; exit 2; }
seen='|'
for shard in "${shards[@]}"; do
  [[ "$shard" =~ ^[0-9][0-9]$ && "$seen" != *"|$shard|"* ]] || { echo 'Invalid or repeated shard' >&2; exit 2; }
  seen+="$shard|"
done

here="$(cd "$(dirname "$0")" && pwd)"
failed=0
for shard in "${shards[@]}"; do
  source_id="full-${track}-${shard}-20260923"
  replay_id="renderer-replay-full-${track}-${shard}-20260923"
  echo "[replay-batch] source=$source_id workers=$workers timeout=$timeout" >&2
  if ! bash "$here/run_replay.sh" "$source_id" "$replay_id" --timeout "$timeout" --workers "$workers"; then
    echo "[replay-batch] failed=$source_id; continuing other verified shards" >&2
    failed=$((failed + 1))
  fi
done
echo "[replay-batch] shards=${#shards[@]} failed=$failed" >&2
[[ "$failed" -eq 0 ]]
