#!/usr/bin/env bash
# Multi-bot runner: start | stop | status
# Env knobs: LOG_DIR, MATCH, RESTART(1/0), DEMO_MODE, SLEEP_BACKOFF

set -Eeuo pipefail
cmd="${1:-start}"
cd "$(dirname "$0")"
shopt -s nullglob

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
PID_FILE="${PID_FILE:-.bots_runner.pid}"

DEMO_MODE="${DEMO_MODE:-true}"
RESTART="${RESTART:-1}"
MATCH="${MATCH:-}"
SLEEP_BACKOFF="${SLEEP_BACKOFF:-5}"

ts() { TZ=UTC date -u '+%Y-%m-%dT%H:%M:%SZ'; }
file_stamp() { date '+%Y%m%d-%H%M%S'; }

bots=(OK_*.py)
IFS=$'\n' bots=($(printf '%s\n' "${bots[@]}" | sort))

if [[ "$cmd" == "stop" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    runner_pid=$(cat "$PID_FILE" || true)
    if [[ -n "${runner_pid:-}" ]] && kill -0 "$runner_pid" 2>/dev/null; then
      echo "[$(ts)] stopping runner pid=$runner_pid…"
      kill -TERM "$runner_pid" || true
      sleep 2
      pkill -P "$runner_pid" || true
    fi
    rm -f "$PID_FILE"
  fi
  # safety: kill any stray bot processes
  pkill -f 'OK_.*\.py' || true
  echo "[$(ts)] all bots stopped."
  exit 0
elif [[ "$cmd" == "status" ]]; then
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[$(ts)] runner is up (pid $(cat "$PID_FILE"))."
  else
    echo "[$(ts)] runner not running."
  fi
  pgrep -fl 'OK_.*\.py' || true
  exit 0
fi

if [[ ${#bots[@]} -eq 0 ]]; then
  echo "[$(ts)] no OK_*.py files found."; exit 1
fi

if [[ -n "$MATCH" ]]; then
  _f=(); for b in "${bots[@]}"; do [[ "$b" =~ $MATCH ]] && _f+=("$b"); done
  bots=("${_f[@]}"); [[ ${#bots[@]} -gt 0 ]] || { echo "[$(ts)] no bots matched MATCH='$MATCH'."; exit 1; }
fi

echo "$$" > "$PID_FILE"

pids=()
cleanup() {
  echo "[$(ts)] stopping all bots…"
  for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${pids[@]:-}"; do kill -KILL "$pid" 2>/dev/null || true; done
  rm -f "$PID_FILE"
}
trap cleanup INT TERM EXIT

start_bot() {
  local bot="$1"
  local log="${LOG_DIR}/${bot%.*}_$(file_stamp).log"
  echo "[$(ts)] starting $bot -> $log"
  (
    while :; do
      echo "[$(ts)] $bot: launch"
      DEMO_MODE="$DEMO_MODE" python3 -u "$bot" >>"$log" 2>&1 || rc=$? || true
      rc=${rc:-0}
      [[ "$RESTART" == "1" ]] || { echo "[$(ts)] $bot: exit rc=$rc; not restarting."; break; }
      echo "[$(ts)] $bot: exited rc=$rc; restarting in ${SLEEP_BACKOFF}s…"
      sleep "$SLEEP_BACKOFF"
    done
  ) &
  pids+=("$!")
}

for bot in "${bots[@]}"; do start_bot "$bot"; done
wait
