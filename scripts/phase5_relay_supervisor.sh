#!/usr/bin/env bash
# Supervise scripts/phase5_relay.py: restart it if it ever exits. Run by SYSTEM
# python (the working side of the conda-env LAN fault). nohup this.
cd "$(dirname "$0")/.."
while true; do
  /usr/bin/python3 scripts/phase5_relay.py
  echo "[supervisor] relay exited rc=$? - restarting in 3s $(date -Iseconds)"
  sleep 3
done
