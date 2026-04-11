#!/usr/bin/env bash
# Run the hidden-state cache pass in parallel across both NUMA nodes.
#
# Each worker handles a disjoint range of sequences from data/tokens.bin,
# writes its shard to data/hidden_<node>.bin, and worker 0 additionally
# saves the backbone's lm_head weight (needed at train time).
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=""

CORES_PER_SOCKET=$(lscpu | awk '/^Core\(s\) per socket:/ {print $4}')
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CORES_PER_SOCKET:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export KMP_AFFINITY="${KMP_AFFINITY:-granularity=fine,compact,1,0}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"
export DNNL_PRIMITIVE_CACHE_CAPACITY="${DNNL_PRIMITIVE_CACHE_CAPACITY:-1024}"

SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BIN_PATH="${BIN_PATH:-data/tokens.bin}"

# Count total sequences in the token bin (uint32 = 4 bytes; we need seq_len+1 per sample).
FILE_BYTES=$(stat -c%s "$BIN_PATH")
NUM_TOKENS=$((FILE_BYTES / 4))
NUM_SEQS=$(( (NUM_TOKENS - 1) / SEQ_LEN ))
HALF=$(( (NUM_SEQS + 1) / 2 ))

echo "[cache_parallel] seq_len=$SEQ_LEN  bs=$BATCH_SIZE  total_seqs=$NUM_SEQS  split_at=$HALF"
echo "[cache_parallel] OMP_NUM_THREADS=$OMP_NUM_THREADS per worker"

LOG0=data/cache_node0.log
LOG1=data/cache_node1.log
mkdir -p data

# Worker 0: NUMA node 0, sequences [0, HALF). Also saves the lm_head weight.
numactl --cpunodebind=0 --membind=0 \
    python cache_hidden.py \
        --bin_path "$BIN_PATH" \
        --seq_len "$SEQ_LEN" \
        --batch_size "$BATCH_SIZE" \
        --start 0 --end "$HALF" \
        --out data/hidden_0.bin \
        --lm_head_out data/lm_head.pt \
    > "$LOG0" 2>&1 &
PID0=$!

# Worker 1: NUMA node 1, sequences [HALF, NUM_SEQS).
numactl --cpunodebind=1 --membind=1 \
    python cache_hidden.py \
        --bin_path "$BIN_PATH" \
        --seq_len "$SEQ_LEN" \
        --batch_size "$BATCH_SIZE" \
        --start "$HALF" --end "$NUM_SEQS" \
        --out data/hidden_1.bin \
    > "$LOG1" 2>&1 &
PID1=$!

echo "[cache_parallel] worker 0 PID=$PID0  log=$LOG0"
echo "[cache_parallel] worker 1 PID=$PID1  log=$LOG1"

# Tail both logs while waiting.
tail -F "$LOG0" "$LOG1" &
TAIL=$!

wait $PID0
RC0=$?
wait $PID1
RC1=$?
kill $TAIL 2>/dev/null || true

if [[ $RC0 -ne 0 || $RC1 -ne 0 ]]; then
    echo "[cache_parallel] FAILED (rc0=$RC0 rc1=$RC1)"
    exit 1
fi

echo "[cache_parallel] concatenating shards -> data/hidden.bin"
cat data/hidden_0.bin data/hidden_1.bin > data/hidden.bin
rm -f data/hidden_0.bin data/hidden_1.bin
ls -lh data/hidden.bin data/lm_head.pt
echo "[cache_parallel] done"
