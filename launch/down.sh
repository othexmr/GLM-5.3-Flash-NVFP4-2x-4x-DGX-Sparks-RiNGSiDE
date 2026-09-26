#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Stop a profile started by launch/up.sh: `docker compose down` in each rank's project, rank 0 (the API) first, then
# the workers. Containers are removed; images, caches, models and the rendered projects stay.
#
#   launch/down.sh --profile tp4|tp2 --site site.env [--go]
#
# Without --go (the default) it prints the commands and runs nothing. It reaches the nodes with ssh (BatchMode) and
# never changes host settings.
set -euo pipefail

profile='' site='' go=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) profile="${2:-}"; shift 2 ;;
    --site) site="${2:-}"; shift 2 ;;
    --go) go=1; shift ;;
    -h|--help) sed -n '4,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "down.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$profile" in tp4|tp2) ;; *) echo "down.sh: --profile tp4 or tp2 required" >&2; exit 2 ;; esac
[ -n "$site" ] && [ -f "$site" ] || { echo "down.sh: --site FILE required (see site.env.example)" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH=(ssh -o BatchMode=yes)
# Shell words for display and for the remote shell: plain when safe, otherwise single-quoted.
word() {
  case "$1" in
    '' | *[!A-Za-z0-9_./:=@,+%-]*)
      local rest="$1" out=''
      while [[ "$rest" == *"'"* ]]; do out+="${rest%%"'"*}'\\''"; rest="${rest#*"'"}"; done
      printf "'%s'" "$out$rest" ;;
    *) printf '%s' "$1" ;;
  esac
}
q() { local out='' w; for w in "$@"; do out+="$(word "$w") "; done; printf '%s' "${out% }"; }

[ "$go" = 1 ] || printf '%s\n' "# Dry run (no --go): nothing runs on any node."
plan="$(python3 -B "$here/compose.py" --profile "$profile" --site "$site" --plan)"
target=() remote_dir=()
while IFS=$'\t' read -r kind a b _c d _rest; do
  if [ "$kind" = rank ]; then target[a]="$b"; remote_dir[a]="$d"; fi
done <<<"$plan"
ranks=${#target[@]}

status=0
for ((i = 0; i < ranks; i++)); do
  r=$i  # rank 0 first: the API stops before its workers
  cmd="cd $(q "${remote_dir[r]}") && docker compose down"
  printf '+ %s\n' "$(q "${SSH[@]}" "${target[r]}" "$cmd")"
  if [ "$go" = 1 ] && ! "${SSH[@]}" "${target[r]}" "$cmd"; then
    echo "down.sh: rank $r on ${target[r]}: docker compose down failed; continuing with the other ranks" >&2
    status=1
  fi
done
exit "$status"
