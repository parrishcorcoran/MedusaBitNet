#!/usr/bin/env bash
# HP Z8 G4 launch script: Medusa heads on frozen BitNet b1.58, CPU-only.
#
# NUMA pinning: the Z8 G4 is a dual-socket box. Cross-socket memory traffic
# on this kind of workload is a real killer (UPI latency + cache ping-pong),
# so we hard-pin the process to NUMA node 0 — both its CPU scheduling and
# its memory allocations.
#
# To run on the *other* socket instead, swap both flags to node 1:
#   numactl --cpunodebind=1 --membind=1 python train.py ...
# (Use `numactl --hardware` to confirm the node layout on your machine.)

set -euo pipefail

cd "$(dirname "$0")"

# Belt-and-suspenders: make absolutely sure no CUDA context is created.
export CUDA_VISIBLE_DEVICES=""

# Intel / oneDNN knobs for AVX-512 on Cascade Lake Xeons.
# One socket on the Z8 G4 is typically ~18-28 physical cores; tune to match.
# `nproc --all` gives logical count; we divide by 2 for physical, then halve
# again if hyperthreading hurts (it usually does for bf16 GEMM).
CORES_PER_SOCKET=$(lscpu | awk '/^Core\(s\) per socket:/ {print $4}')
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CORES_PER_SOCKET:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export KMP_AFFINITY="${KMP_AFFINITY:-granularity=fine,compact,1,0}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"
export DNNL_PRIMITIVE_CACHE_CAPACITY="${DNNL_PRIMITIVE_CACHE_CAPACITY:-1024}"
# Force bf16-friendly math and disable any stray GPU probes.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"

echo "[run_z8] OMP_NUM_THREADS=$OMP_NUM_THREADS  CORES_PER_SOCKET=$CORES_PER_SOCKET"
echo "[run_z8] pinning to NUMA node 0"

# If the first arg is --benchmark, run the smoke-test / timing script instead
# of the full training loop. Everything after --benchmark is forwarded to it.
#   ./run_z8_training.sh --benchmark
#   ./run_z8_training.sh --benchmark --bench_steps 20 --seq_len 1024
SCRIPT="train.py"
if [[ "${1:-}" == "--benchmark" ]]; then
    SCRIPT="benchmark.py"
    shift
    echo "[run_z8] running benchmark smoke-test"
fi

# --cpunodebind=0  -> schedule threads only on socket 0's cores
# --membind=0      -> allocate memory only from socket 0's DIMMs
exec numactl --cpunodebind=0 --membind=0 \
    python "$SCRIPT" "$@"
