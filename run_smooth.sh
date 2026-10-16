#!/bin/bash
# Live face bridge, 'smooth' preset. Start TouchDesigner (td/talking.toe) first, then run this. Ctrl+C to stop.
cd "$(dirname "$0")"
pkill -f "live/bridge.py" 2>/dev/null
PYTHONPATH=live exec ./.venv/bin/python -u live/bridge.py --preset smooth "$@"
