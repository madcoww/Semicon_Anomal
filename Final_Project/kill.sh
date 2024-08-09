#!/bin/bash

# PID 파일이 존재하는지 확인
if [ -f pid.file ]; then
    # PID 파일에서 PID를 읽어와 종료
    PID=$(cat pid.file)
    kill $PID
    # 종료가 성공했는지 확인
    if [ $? -eq 0 ]; then
        echo "Script with PID $PID has been killed."
        # PID 파일 삭제
        rm pid.file
    else
        echo "Failed to kill script with PID $PID."
    fi
else
    echo "PID file not found. Is the script running?"
fi
