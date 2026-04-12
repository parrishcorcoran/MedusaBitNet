#!/bin/bash
# Measure power during BitNet inference using RAPL
set -euo pipefail

RAPL=/sys/class/powercap/intel-rapl:0/energy_uj
MODEL="$1"
BINARY="$2"
PROMPT="${3:-The meaning of life is to understand the nature of consciousness and explore}"
N_TOKENS="${4:-256}"
THREADS="${5:-16}"
LABEL="${6:-test}"

echo "=== Power Benchmark: $LABEL ==="
echo "Model: $MODEL"
echo "Tokens: $N_TOKENS, Threads: $THREADS"

# Idle baseline (2 seconds)
E0=$(cat $RAPL)
sleep 2
E1=$(cat $RAPL)
IDLE_UJ=$((E1 - E0))
IDLE_W=$(echo "scale=2; $IDLE_UJ / 2000000" | bc)
echo "Idle power: ${IDLE_W}W"

# Run inference with power measurement
E_START=$(cat $RAPL)
T_START=$(date +%s%N)

$BINARY -m "$MODEL" -p "$PROMPT" -n $N_TOKENS -t $THREADS --temp 0 --no-display-prompt 2>&1 | \
    grep -E "eval time|total time|prompt eval" > /tmp/bench_perf.txt

T_END=$(date +%s%N)
E_END=$(cat $RAPL)

# Calculate
DURATION_NS=$((T_END - T_START))
DURATION_S=$(echo "scale=3; $DURATION_NS / 1000000000" | bc)
ENERGY_UJ=$((E_END - E_START))
ENERGY_J=$(echo "scale=3; $ENERGY_UJ / 1000000" | bc)
AVG_POWER_W=$(echo "scale=2; $ENERGY_J / $DURATION_S" | bc)
INFERENCE_POWER_W=$(echo "scale=2; $AVG_POWER_W - $IDLE_W" | bc)

echo ""
echo "Duration: ${DURATION_S}s"
echo "Total energy: ${ENERGY_J}J"
echo "Average power (total): ${AVG_POWER_W}W"
echo "Idle power: ${IDLE_W}W"
echo "Inference power (above idle): ${INFERENCE_POWER_W}W"
echo ""
cat /tmp/bench_perf.txt
echo ""

# Extract tok/s from perf output
TOK_S=$(grep "eval time" /tmp/bench_perf.txt | tail -1 | grep -oP '[\d.]+(?= tokens per second)')
if [ -n "$TOK_S" ]; then
    TOKENS_PER_JOULE=$(echo "scale=2; $TOK_S / $AVG_POWER_W" | bc)
    TOKENS_PER_WATT=$(echo "scale=2; $TOK_S / $AVG_POWER_W" | bc)
    echo "Tokens/sec: $TOK_S"
    echo "Tokens/joule: $TOKENS_PER_JOULE"
    echo "Tokens/watt: $TOKENS_PER_WATT (= tok/s per watt)"
fi
echo "=== End $LABEL ==="
