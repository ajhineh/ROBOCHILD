#!/bin/bash
# اسکریپت ریستارت ربات
pkill -f "src.main" 2>/dev/null || true
sleep 4
cd /home/ubuntu/opt/ROBOCHILD
nohup venv/bin/python -m src.main >> robochild_x.log 2>&1 &
echo "Bot started with PID: $!"
sleep 6
ps aux | grep "src.main" | grep -v grep | awk '{print "RUNNING pid=" $2 " mem=" $6 "KB"}'
