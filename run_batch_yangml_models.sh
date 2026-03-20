#!/usr/bin/env bash
set -euo pipefail

# ---- Config (edit as needed) ----
RUN_MODE="${RUN_MODE:-crossval}"               # crossval | test
LOG_DIR="${LOG_DIR:-logs_local}"
PYTHON_BIN="${PYTHON_BIN:-python}"            # or path to your venv/bin/python
VENV_ACTIVATE="${VENV_ACTIVATE:-}"            # e.g. /home/user/venv/bin/activate

# One process per (model,dataset) — limit inner threading
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

MODELS=(
  "BernoulliNB_Yang2018"
  "LogisticRegressionL1_Yang2018"
  "LogisticRegressionL2_Yang2018"
  "RandomForest_Yang2018"
  "SVCLinear_Yang2018"
  "SVCRBF_Yang2018"
  # "SVCLinearOptunity_Yang2018"
  # "SVCRBFOptunity_Yang2018"
)

DATASETS=(
  "_All_SNPs/"
  "_CodingRegion_SNPs/"
  "_Tier12_Genes/"
  "_Tier12_Genes_without10/"
)

mkdir -p "$LOG_DIR" saved_models

# optional venv
if [[ -n "$VENV_ACTIVATE" && -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

run_one() {
  local model="$1" dataset="$2"
  local tag="${model}_$(echo "$dataset" | tr -d '/')"  # e.g., SVCRBF_Yang2018__All_SNPs
  echo "[$(date +%F_%T)] START  $tag"
  "${PYTHON_BIN}" main_v2.py -s "$dataset" -m "$model" -r "$RUN_MODE" \
    >"${LOG_DIR}/${tag}.out" 2>"${LOG_DIR}/${tag}.err"
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    echo "[$(date +%F_%T)] SUCCESS $tag"
  else
    echo "[$(date +%F_%T)] FAILED  $tag (exit code: $exit_code)"
  fi
  return $exit_code
}

# Launch ALL model+dataset pairs in parallel
echo "=== Launching parallel jobs ==="
echo "Models: ${#MODELS[@]}, Datasets: ${#DATASETS[@]}"
echo "Total jobs: $((${#MODELS[@]} * ${#DATASETS[@]}))"
echo ""

pids=()
job_names=()

# Swap loop order to iterate datasets first (outer), models second (inner)
for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    # Launch in background immediately
    run_one "$model" "$ds" &
    pids+=("$!")
    job_names+=("${model} + ${ds}")
    # Small delay to avoid overwhelming the system at launch
    sleep 0.1
  done
done

echo "All ${#pids[@]} jobs launched. Waiting for completion..."
echo ""

# Wait for all background jobs and track failures
fail=0
failed_jobs=()
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "$pid"; then
    fail=1
    failed_jobs+=("${job_names[$i]}")
  fi
done

# Summary
echo ""
echo "=== Execution Summary ==="
if [[ $fail -eq 0 ]]; then
  echo "✓ All ${#pids[@]} jobs finished successfully!"
else
  echo "✗ Some jobs failed. Check logs in ${LOG_DIR}/"
  echo ""
  echo "Failed jobs (${#failed_jobs[@]}):"
  for job in "${failed_jobs[@]}"; do
    echo "  - $job"
  done
  exit 1
fi
