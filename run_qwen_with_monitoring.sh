#!/bin/bash
# Run score_sites_qwen.py with hardware monitoring - simplified version

LOG_FILE="hardware_metrics_qwen_runraw.md"
MONITOR_LOG="/tmp/gpu_monitor_$$.log"

echo "Starting hardware monitoring..."
echo "Log file: $LOG_FILE"

# Capture pre-run GPU state
echo "## Pre-run GPU State" > "$LOG_FILE"
nvidia-smi --query-gpu=name,memory.total,driver_version,temperature.gpu,utilization.gpu,clocks.current.sm,clocks.current.memory --format=csv,noheader,nounits 2>&1 | head -1 >> "$LOG_FILE" || true
echo "" >> "$LOG_FILE"

# Capture pre-run CPU thermal
echo "## Pre-run CPU Thermal" >> "$LOG_FILE"
sensors 2>/dev/null | grep -E "acpitz" | head -8 >> "$LOG_FILE" || echo "sensors not available" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Start GPU monitoring in background (every 2 seconds)
(
    echo "Timestamp,Temp(C),GPU_Util(%),Mem_Util(%),Clock(MHz)"
    while true; do
        TS=$(date +%s)
        STATS=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory,clocks.current.sm --format=csv,noheader,nounits 2>/dev/null | sed 's/, /,/g')
        echo "$TS,$STATS"
        sleep 2
    done
) > "$MONITOR_LOG" &
MONITOR_PID=$!
echo "Monitoring PID: $MONITOR_PID"

# Run the script
echo "## Running score_sites_qwen.py..." >> "$LOG_FILE"
START_TIME=$(date +%s)
cd /home/acergn100_6/smart-city-management
python3 AI/score_sites_qwen.py 2>&1 | tee -a "$LOG_FILE"
END_TIME=$(date +%s)
RUN_TIME=$((END_TIME - START_TIME))

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true
sleep 1

# Capture post-run GPU state
echo "" >> "$LOG_FILE"
echo "## Post-run GPU State" >> "$LOG_FILE"
nvidia-smi --query-gpu=name,memory.total,driver_version,temperature.gpu,utilization.gpu,clocks.current.sm,clocks.current.memory --format=csv,noheader,nounits 2>&1 | head -1 >> "$LOG_FILE" || true
echo "" >> "$LOG_FILE"

# Capture post-run CPU thermal
echo "## Post-run CPU Thermal" >> "$LOG_FILE"
sensors 2>/dev/null | grep -E "acpitz" | head -8 >> "$LOG_FILE" || echo "sensors not available" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Process GPU utilization data
echo "## GPU Utilization During Run (sampled every 2s)" >> "$LOG_FILE"
echo "Timestamp,Temp(C),GPU_Util(%),Mem_Util(%),Clock(MHz)" >> "$LOG_FILE"

if [ -f "$MONITOR_LOG" ] && [ -s "$MONITOR_LOG" ]; then
    FIRST_TS=$(head -2 "$MONITOR_LOG" | tail -1 | cut -d',' -f1)
    if [ -z "$FIRST_TS" ] || ! [[ "$FIRST_TS" =~ ^[0-9]+$ ]]; then
        FIRST_TS=0
    fi
    tail -n +2 "$MONITOR_LOG" | while read line; do
        TS=$(echo "$line" | cut -d',' -f1)
        if [ -n "$TS" ] && [[ "$TS" =~ ^[0-9]+$ ]]; then
            REL_TS=$((TS - FIRST_TS))
            echo "$REL_TS,${line#*,}" >> "$LOG_FILE"
        fi
    done
fi
echo "" >> "$LOG_FILE"

# Summary
echo "## Summary" >> "$LOG_FILE"
echo "Total runtime: ${RUN_TIME}s" >> "$LOG_FILE"

if [ -f "$MONITOR_LOG" ] && [ -s "$MONITOR_LOG" ]; then
    TEMPS=$(tail -n +2 "$MONITOR_LOG" | cut -d',' -f2 | sort -n | tr '\n' ' ')
    UTILS=$(tail -n +2 "$MONITOR_LOG" | cut -d',' -f3 | tr -d ' %' | sort -n | tr '\n' ' ')
    PEAK_TEMP=$(echo "$TEMPS" | awk '{print $NF}')
    PEAK_UTIL=$(echo "$UTILS" | awk '{print $NF}')
    echo "Peak GPU temp: ${PEAK_TEMP}C" >> "$LOG_FILE"
    echo "Peak GPU util: ${PEAK_UTIL}%" >> "$LOG_FILE"
fi

rm -f "$MONITOR_LOG"
echo "" >> "$LOG_FILE"
echo "Hardware metrics saved to: $LOG_FILE"