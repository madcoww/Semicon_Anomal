#!/bin/bash

PYTHON_EXEC="/SSD/ai_test/.venv/bin/python3.11"
SCRIPT_NAME="/SSD/ai_test/AlbertMain.py"

nohup "$PYTHON_EXEC" "$SCRIPT_NAME" > output.log 2>&1 &
echo $! > pid.file
echo "Script is running in the background with PID $(cat pid.file)"
