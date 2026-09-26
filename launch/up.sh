#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Start a profile on its nodes with Docker Compose, in the documented order, and wait for the API.
#
#   launch/up.sh --profile tp4|tp2 --site site.env [--build] [--timeout SECONDS] [--go]
#
# Without --go (the default) this is a dry run: it renders the compose projects into a temporary directory and prints
# every command it would run; nothing runs on any node. With --go it runs them:
#   1. renders deploy/<profile>/rank<N>/{compose.yaml,.env} with launch/compose.py (site values from the site file);
#   2. with --build: builds the profile image on rank 0's node (`docker compose --profile build build`: the vLLM source
#      image, then docker/Dockerfile's profile target) and copies it to every other node with docker save | docker load,
#      from rank 0's node directly when SWITCHLESS_SHARE_TARGETS is set, otherwise through this machine;
#   3. checks that every node has the same image ID as rank 0's node;
#   4. per rank, workers first and rank 0 last: checks the model and drafter directories, creates the cache and profile
#      directories of the site file, copies the rank's project to SWITCHLESS_REMOTE_REPO/deploy/<profile>/rank<N> and
#      runs `docker compose up -d --no-build --pull never glm53`;
#   5. waits until rank 0 answers /health and /v1/models lists the model, and prints the endpoint.
# It reaches the nodes with ssh (BatchMode) and never changes host settings (sysctl, page cache, firewall, network).
# Without --build it pulls and builds nothing. Every node needs a clone of this repository at SWITCHLESS_REMOTE_REPO
# (the compose build context). docker/README.md and docs/operations.md explain the steps.
set -euo pipefail

usage() {
  sed -n '4,22p' "$0" | sed 's/^# \{0,1\}//'
}

profile='' site='' go=0 build=0 timeout=7200
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) profile="${2:-}"; shift 2 ;;
    --site) site="${2:-}"; shift 2 ;;
    --build) build=1; shift ;;
    --timeout) timeout="${2:-}"; shift 2 ;;
    --go) go=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "up.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$profile" in tp4|tp2) ;; *) echo "up.sh: --profile tp4 or tp2 required" >&2; exit 2 ;; esac
[ -n "$site" ] && [ -f "$site" ] || { echo "up.sh: --site FILE required (see site.env.example)" >&2; exit 2; }
case "$timeout" in ''|*[!0-9]*) echo "up.sh: --timeout takes seconds" >&2; exit 2 ;; esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
SSH=(ssh -o BatchMode=yes)

say() { printf '%s\n' "$*"; }
# Print a command; run it only with --go.
run() {
  printf '+ %s\n' "$(q "$@")"
  if [ "$go" = 1 ]; then "$@"; fi
}
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

if [ "$go" = 1 ]; then
  out="$root/deploy/$profile"
else
  out="$(mktemp -d)"
  trap 'rm -rf "$out"' EXIT
  say "# Dry run (no --go): nothing runs on any node. Rendered files go to a temporary directory."
fi
python3 -B "$here/compose.py" --profile "$profile" --site "$site" --out "$out" >/dev/null
plan="$(python3 -B "$here/compose.py" --profile "$profile" --site "$site" --out "$out" --plan)"

image='' remote='' port='' model='' api_host=''
target=() local_dir=() remote_dir=() container=() jit=() prof=() hf=() model_dir=() drafter=() share=()
while IFS=$'\t' read -r kind a b c d e f g h i j k; do
  case "$kind" in
    image) image="$a" ;;
    remote_repo) remote="$a" ;;
    api_port) port="$a" ;;
    model_name) model="$a" ;;
    api_host) api_host="$a" ;;
    rank)
      target[a]="$b"; local_dir[a]="$c"; remote_dir[a]="$d"; container[a]="$e"; jit[a]="$f"; prof[a]="$g"
      hf[a]="$h"; model_dir[a]="$i"; drafter[a]="$j"; share[a]="$k" ;;
  esac
done <<<"$plan"
ranks=${#target[@]}
say "# Profile $profile: $ranks ranks, image $image, repository on the nodes: $remote"

image_id() { "${SSH[@]}" "$1" "$(q docker image inspect --format '{{.Id}}' "$image")" 2>/dev/null || true; }

copy_project() {  # rank
  local r="$1"
  run "${SSH[@]}" "${target[r]}" "$(q mkdir -p "${remote_dir[r]}")"
  run scp -q "${local_dir[r]}/compose.yaml" "${local_dir[r]}/.env" "${target[r]}:${remote_dir[r]}/"
}

if [ "$build" = 1 ]; then
  say "# Build the profile image on rank 0's node (docker/README.md: an estimated 1.5-3 hours the first time)"
  copy_project 0
  run "${SSH[@]}" "${target[0]}" "cd $(q "${remote_dir[0]}") && docker compose --profile build build"
  for ((r = 1; r < ranks; r++)); do
    say "# Share the image with rank $r (skipped when it already has rank 0's image ID)"
    if [ "$go" = 1 ] && [ -n "$(image_id "${target[r]}")" ] && [ "$(image_id "${target[r]}")" = "$(image_id "${target[0]}")" ]; then
      say "# rank $r already has $image"
      continue
    fi
    if [ "${share[r]}" != "-" ]; then
      run "${SSH[@]}" "${target[0]}" "$(q docker save "$image") | $(q ssh -o BatchMode=yes "${share[r]}" docker load)"
    else
      printf '+ %s | %s\n' "$(q "${SSH[@]}" "${target[0]}" "$(q docker save "$image")")" \
        "$(q "${SSH[@]}" "${target[r]}" docker load)"
      if [ "$go" = 1 ]; then
        "${SSH[@]}" "${target[0]}" "$(q docker save "$image")" | "${SSH[@]}" "${target[r]}" docker load
      fi
    fi
  done
fi

say "# Every node must hold the same image"
run "${SSH[@]}" "${target[0]}" "$(q docker image inspect --format '{{.Id}}' "$image")"
if [ "$go" = 1 ]; then
  want="$(image_id "${target[0]}")"
  [ -n "$want" ] || { echo "up.sh: rank 0 has no image $image: build it (--build, docker/README.md)" >&2; exit 1; }
  for ((r = 1; r < ranks; r++)); do
    got="$(image_id "${target[r]}")"
    [ "$got" = "$want" ] || { echo "up.sh: rank $r has ${got:-no image} for $image, rank 0 has $want: share it (--build or docker/README.md)" >&2; exit 1; }
  done
fi

start_rank() {  # rank
  local r="$1"
  say "# Rank $r on ${target[r]} (container ${container[r]})"
  run "${SSH[@]}" "${target[r]}" "$(q test -d "${model_dir[r]}") && $(q test -d "${drafter[r]}/snapshots")"
  run "${SSH[@]}" "${target[r]}" "$(q mkdir -p "${hf[r]}" "${jit[r]}" "${prof[r]}")"
  copy_project "$r"
  run "${SSH[@]}" "${target[r]}" "cd $(q "${remote_dir[r]}") && docker compose up -d --no-build --pull never glm53"
}
for ((r = ranks - 1; r >= 1; r--)); do start_rank "$r"; done
start_rank 0

health="$(q curl -fsS -m 10 "http://127.0.0.1:${port}/health")"
models="$(q curl -fsS -m 30 "http://127.0.0.1:${port}/v1/models")"
say "# Wait for rank 0: the first boot compiles kernels and runs the startup warm-up before the API listens"
printf '+ %s   (every 30 s, at most %s s; every rank'"'"'s container must keep running)\n' \
  "$(q "${SSH[@]}" "${target[0]}" "$health")" "$timeout"
if [ "$go" = 1 ]; then
  start=$SECONDS
  until "${SSH[@]}" "${target[0]}" "$health" >/dev/null 2>&1; do
    for ((r = 0; r < ranks; r++)); do
      state="$("${SSH[@]}" "${target[r]}" "$(q docker inspect --format '{{.State.Running}}' "${container[r]}")" 2>/dev/null || true)"
      if [ "$state" != "true" ]; then
        echo "up.sh: rank $r (${container[r]} on ${target[r]}) is not running; see: docker logs ${container[r]}" >&2
        exit 1
      fi
    done
    if [ $((SECONDS - start)) -ge "$timeout" ]; then
      echo "up.sh: rank 0 did not answer /health within ${timeout} s; see: docker logs ${container[0]}" >&2
      exit 1
    fi
    sleep 30
  done
fi
run "${SSH[@]}" "${target[0]}" "$models"
if [ "$go" = 1 ]; then
  "${SSH[@]}" "${target[0]}" "$models" | grep -q -- "\"$model\"" \
    || { echo "up.sh: /v1/models does not list $model" >&2; exit 1; }
fi

say "# Endpoint: http://${api_host}:${port}/v1 (OpenAI-compatible, model $model)"
say "# Check it with a completion request (docs/operations.md). No authentication and no TLS: keep the port on a"
say "# trusted network or behind an authenticating TLS proxy. Stop the profile with launch/down.sh."
