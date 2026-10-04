#!/bin/bash
# Stages of the GPU test of the denoiser's stopping rule on gautschi.
#
# Usage, inside an allocation of H100s with 14 cores per GPU:
#     bash mace4d_stopping.sh <stage>
# Each stage appends its output to $RUNS/logs/<stage>.log.  The stages:
#   setup_c   make wt_C (greg_dev plus gradient_rule.patch) and venv_C; runs on the
#             login node, because it neither imports torch nor needs a GPU
#   smoke     the stages below at a small size (12 frames, downsampling 4, 2 iterations)
#   init      compute and cache the initial image that every run starts from
#   sigma     option 2's noise level from the initial image
#   A B C D   the MACE4D runs of versions A to D (mace4d_stopping.md); D needs sigma
#   Bw Cw     versions B and C with the denoiser warm start on
#   frame     denoise on one frame of the initial image, for A to D, on one GPU; D needs sigma
#   calls     the distance of the sampled calls from their MAP estimates, one
#             process per GPU
#   compare   the comparison of the runs
#   collect   copy the small result files to $ROOT/collect/mace4d_stopping
#   core      init sigma A B C D frame calls compare collect, in order
#   extra     Bw Cw calls compare collect, in order
#   all       core, then extra
#   core_from <stage>, extra_from <stage>
#             the stages of core or of extra from <stage> on, to resume after a failure
# The composite stages stop at the first stage that fails.  The setup of $ROOT
# (src, plans, wt_A, wt_B, venv_A, venv_B) is in mace4d_stopping.md.  On another
# cluster, set DENOISER_STOP_ROOT to the folder that holds that setup.

set -eo pipefail
STAGE=${1:?give a stage}

ROOT=${DENOISER_STOP_ROOT:-/scratch/gautschi/buzzard/denoiser_stop}
EXP=$ROOT/plans/plans/improve_denoiser/experiments
DATA=/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/
RUNS=$ROOT/runs
mkdir -p "$RUNS/logs"
exec >> "$RUNS/logs/$STAGE.log" 2>&1
echo "=== stage $STAGE started $(date) on $(hostname)"

CORE=(init sigma A B C D frame calls compare collect)
EXTRA=(Bw Cw calls compare collect)

composite() {             # composite <stage> ...: run the stages in order, stopping at a failure
    local stage
    for stage in "$@"; do
        echo "starting $stage $(date)"
        bash "$0" "$stage" || { echo "stage $stage failed $(date)"; exit 1; }
        echo "finished $stage $(date)"
    done
}

from_stage() {            # from_stage <first> <stage> ...: run the stages from <first> on
    local first=$1 stage found=0 todo=()
    shift
    for stage in "$@"; do
        [ "$stage" = "$first" ] && found=1
        [ "$found" -eq 1 ] && todo+=("$stage")
    done
    [ "$found" -eq 1 ] || { echo "no stage $first in: $*"; exit 1; }
    composite "${todo[@]}"
}

case $STAGE in
core)       composite "${CORE[@]}"; exit 0 ;;
extra)      composite "${EXTRA[@]}"; exit 0 ;;
all)        composite core extra; exit 0 ;;
core_from)  from_stage "${2:?give the first stage}" "${CORE[@]}"; exit 0 ;;
extra_from) from_stage "${2:?give the first stage}" "${EXTRA[@]}"; exit 0 ;;
esac

# The node preamble sets OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1, which leave torch
# one host thread; unsetting both gives torch the cores of the allocation.
set +e
source ~/load_conda_cuda.sh
set -e
unset OMP_NUM_THREADS MKL_NUM_THREADS
nvidia-smi --query-gpu=index,name,memory.used --format=csv || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
cd "$RUNS"                # no folder named mbirtorch here, so the package is not shadowed

py() {                    # py <version> <script> <args>: run a script of the experiment folder
    local version=$1 script=$2
    shift 2
    # The scripts stop unless mbirtorch is imported from this worktree.
    EXPECT_LIBRARY=$ROOT/wt_$version "$ROOT/venv_$version/bin/python" -u "$EXP/$script" "$@"
}

mace() {                  # mace <label> <version> <run folder> <init folder> <args>
    local label=$1 version=$2 dir=$3 init=$4
    shift 4
    # Each run compiles with its own empty caches, so every run pays the same compile
    # time.  The smoke stage sets CACHE_ROOT to share one cache per version.
    local cache=${CACHE_ROOT:+$CACHE_ROOT/$version}
    cache=${cache:-$dir/cache}
    export TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton
    py "$version" mace4d_stopping.py --data_path "$DATA" --run_dir "$dir" --init_dir "$init" \
        --label "$label" "$@"
}

robust_sigma() {          # robust_sigma <init folder>: write robust_sigma.json there and print it
    py C robust_sigma.py --init "$1/init_recon.npy" --out "$1/robust_sigma.json" > /dev/null
    cat "$1/robust_sigma.json"
    echo
}

sigma_of() {              # sigma_of <init folder>: print option 2's noise level
    "$ROOT/venv_C/bin/python" -c "import json; print(json.load(open('$1/robust_sigma.json'))['robust'])"
}

calls() {                 # calls <out folder> <cache folder> <run folder> ...: one process per GPU
    local out=$1 cache=$2
    shift 2
    local gpus k pid status=0 pids=()
    IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-0}"
    rm -rf "$out"
    mkdir -p "$out"
    for k in "${!gpus[@]}"; do
        CUDA_VISIBLE_DEVICES=${gpus[$k]} py C mace4d_calls.py --runs "$@" --shard "$k/${#gpus[@]}" \
            --out "$out/shard_$k" --cache "$cache" > "$out/shard_$k.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    tail -n 3 "$out"/shard_*.log
    [ "$status" -eq 0 ] || { echo "a calls shard failed; see $out/shard_*.log"; return 1; }
    py C mace4d_calls.py --merge "$out"/shard_*/ --out "$out"
}

case $STAGE in
setup_c)
    # venv_C uses the same base interpreter as venv_A and venv_B.
    BASE=$(sed -n 's/^executable = //p' "$ROOT/venv_A/pyvenv.cfg")
    echo "base interpreter: $BASE"
    [ -x "$BASE" ] || { echo "venv_A/pyvenv.cfg names no base interpreter"; exit 1; }
    git -C "$ROOT/src" worktree add --detach "$ROOT/wt_C" 5f62506
    git -C "$ROOT/wt_C" apply "$EXP/gradient_rule.patch"
    git -C "$ROOT/wt_C" diff --stat
    "$BASE" -m venv --system-site-packages "$ROOT/venv_C"
    # The compat form, because the base environment holds a regular install of mbirtorch.
    "$ROOT/venv_C/bin/pip" install -e "$ROOT/wt_C" --no-deps --config-settings editable_mode=compat
    "$ROOT/venv_C/bin/python" -c "import importlib.util; print(importlib.util.find_spec('mbirtorch').origin)"
    ;;
smoke)
    SMOKE=$RUNS/smoke
    export CACHE_ROOT=$SMOKE/cache
    small=(--num_frames 12 --downsampling 4 --max_mace_iterations 2 --sample_iterations 1,2)
    mace init B "$SMOKE/init_run" "$SMOKE/init" --num_frames 12 --downsampling 4 --max_mace_iterations 0
    robust_sigma "$SMOKE/init"
    mace A A "$SMOKE/A" "$SMOKE/init" "${small[@]}"
    mace B B "$SMOKE/B" "$SMOKE/init" "${small[@]}"
    mace C C "$SMOKE/C" "$SMOKE/init" "${small[@]}" --rule gradient
    mace D C "$SMOKE/D" "$SMOKE/init" "${small[@]}" --rule gradient --sigma_noise "$(sigma_of "$SMOKE/init")"
    mace Cw C "$SMOKE/Cw" "$SMOKE/init" "${small[@]}" --rule gradient --denoiser_warm_start
    py A frame_check.py --init "$SMOKE/init/init_recon.npy" --out_dir "$SMOKE/frame" --label A
    py C frame_check.py --init "$SMOKE/init/init_recon.npy" --out_dir "$SMOKE/frame" --label D \
        --rule gradient --sigma_noise "$(sigma_of "$SMOKE/init")"
    calls "$SMOKE/calls" "$SMOKE/calls_cache" "$SMOKE/A" "$SMOKE/C" "$SMOKE/Cw" "$SMOKE/frame"
    py C mace4d_compare.py --runs A="$SMOKE/A" B="$SMOKE/B" C="$SMOKE/C" D="$SMOKE/D" Cw="$SMOKE/Cw" \
        --baseline B --out "$SMOKE/compare"
    ;;
init)  mace init B "$RUNS/init_run" "$RUNS/init" --max_mace_iterations 0 ;;
sigma) robust_sigma "$RUNS/init" ;;
A)     mace A A "$RUNS/A" "$RUNS/init" ;;
B)     mace B B "$RUNS/B" "$RUNS/init" ;;
C)     mace C C "$RUNS/C" "$RUNS/init" --rule gradient ;;
D)     mace D C "$RUNS/D" "$RUNS/init" --rule gradient --sigma_noise "$(sigma_of "$RUNS/init")" ;;
Bw)    mace Bw B "$RUNS/Bw" "$RUNS/init" --denoiser_warm_start ;;
Cw)    mace Cw C "$RUNS/Cw" "$RUNS/init" --rule gradient --denoiser_warm_start ;;
frame)
    INIT=$RUNS/init/init_recon.npy
    py A frame_check.py --init "$INIT" --out_dir "$RUNS/frame" --label A
    py B frame_check.py --init "$INIT" --out_dir "$RUNS/frame" --label B
    py C frame_check.py --init "$INIT" --out_dir "$RUNS/frame" --label C --rule gradient
    py C frame_check.py --init "$INIT" --out_dir "$RUNS/frame" --label D --rule gradient \
        --sigma_noise "$(sigma_of "$RUNS/init")"
    ;;
calls)
    runs=()
    for label in A B C D Bw Cw frame; do
        [ -d "$RUNS/$label/samples" ] && runs+=("$RUNS/$label")
    done
    calls "$RUNS/calls" "$RUNS/calls_cache" "${runs[@]}"
    ;;
compare)
    runs=()
    for label in A B C D Bw Cw; do
        [ -f "$RUNS/$label/summary.json" ] && runs+=("$label=$RUNS/$label")
    done
    py C mace4d_compare.py --runs "${runs[@]}" --baseline B --out "$RUNS/compare"
    ;;
collect)
    OUT=$ROOT/collect/mace4d_stopping
    mkdir -p "$OUT"
    for label in init_run A B C D Bw Cw; do
        d=$RUNS/$label
        [ -d "$d" ] || continue
        mkdir -p "$OUT/$label"
        cp "$d/summary.json" "$d/calls.csv" "$d/volumes.csv" "$OUT/$label/" 2>/dev/null || true
        cp "$d/logs/run_info.txt" "$d/logs/timing_log.csv" "$d/logs/task_log.csv" "$OUT/$label/" \
            2>/dev/null || true
    done
    cp "$RUNS"/frame/frame_*.json "$OUT/" 2>/dev/null || true
    cp "$RUNS"/calls/*.csv "$RUNS"/compare/* "$OUT/" 2>/dev/null || true
    cp "$RUNS/init/robust_sigma.json" "$OUT/" 2>/dev/null || true
    mkdir -p "$OUT/logs"
    for f in "$RUNS"/logs/*.log "$RUNS"/calls/shard_*.log; do
        [ -f "$f" ] || continue
        grep -v "Error sino RMSE" "$f" | tail -n 400 > "$OUT/logs/$(basename "$f")" || true
    done
    du -sh "$OUT"
    ;;
*)
    echo "unknown stage $STAGE"
    exit 1
    ;;
esac
echo "=== stage $STAGE finished $(date)"
