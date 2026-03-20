#!/usr/bin/env bash
set -euo pipefail

# ---- Configuration ----
LOG_DIR="${LOG_DIR:-logs_test}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"

# Limit threading for parallel execution
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# ---- Test Datasets ----
TEST_DATASETS=(
  "_Chinese_CodingOnly_test"
  "_Chinese_Codingtier12_10_test"
  "_Chinese_Cutoff10_test"
)

# ---- Training Datasets (that models were trained on) ----
TRAIN_DATASETS=(
  "_All_SNPs"
  "_CodingRegion_SNPs"
  "_Tier12_Genes"
  "_Tier12_Genes_without10"
)

# ---- Models to Test ----
# List all models you want to test (only those with trained models will work)
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

mkdir -p "$LOG_DIR"

# Optional venv activation
if [[ -n "$VENV_ACTIVATE" && -f "$VENV_ACTIVATE" ]]; then
  source "$VENV_ACTIVATE"
fi

run_one_test() {
  local test_dataset="$1"
  local train_dataset="$2"
  local model="$3"
  
  local tag="TEST_${test_dataset}_FROM_${train_dataset}_${model}"
  tag=$(echo "$tag" | tr '/' '_' | tr ' ' '_')
  
  echo "[$(date +%F_%T)] START  $tag"
  
  "${PYTHON_BIN}" main_test.py \
    -t "$test_dataset" \
    -s "$train_dataset" \
    -m "$model" \
    >"${LOG_DIR}/${tag}.out" 2>"${LOG_DIR}/${tag}.err"
  
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    echo "[$(date +%F_%T)] SUCCESS $tag"
  else
    echo "[$(date +%F_%T)] FAILED  $tag (exit code: $exit_code)"
  fi
  return $exit_code
}

# ---- Main Execution ----
echo "========================================"
echo "TB Drug Resistance Model Testing"
echo "========================================"
echo ""
echo "Test Datasets: ${#TEST_DATASETS[@]}"
echo "Train Datasets: ${#TRAIN_DATASETS[@]}"
echo "Models: ${#MODELS[@]}"
echo ""
echo "Total test combinations: $((${#TEST_DATASETS[@]} * ${#TRAIN_DATASETS[@]} * ${#MODELS[@]}))"
echo ""
echo "Note: Only combinations with trained models will succeed."
echo "      Missing models will be logged as failures."
echo ""
echo "========================================"
echo ""

pids=()
job_names=()

# Launch all test combinations in parallel
for test_ds in "${TEST_DATASETS[@]}"; do
  for train_ds in "${TRAIN_DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      run_one_test "$test_ds" "$train_ds" "$model" &
      pids+=("$!")
      job_names+=("Test:${test_ds} | Train:${train_ds} | Model:${model}")
      sleep 0.1  # Small delay to avoid overwhelming the system
    done
  done
done

echo "All ${#pids[@]} test jobs launched. Waiting for completion..."
echo ""

# Wait for all jobs and track failures
fail=0
failed_jobs=()
successful_jobs=0

for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if wait "$pid"; then
    ((successful_jobs++))
  else
    fail=1
    failed_jobs+=("${job_names[$i]}")
  fi
done

# Summary
echo ""
echo "========================================"
echo "Test Execution Summary"
echo "========================================"
echo ""
echo "Total jobs: ${#pids[@]}"
echo "Successful: $successful_jobs"
echo "Failed: ${#failed_jobs[@]}"
echo ""

if [[ $fail -eq 0 ]]; then
  echo "✓ All test jobs completed successfully!"
else
  echo "⚠ Some test jobs failed (likely due to missing trained models)"
  echo ""
  echo "Failed jobs (${#failed_jobs[@]}):"
  for job in "${failed_jobs[@]}"; do
    echo "  - $job"
  done
  echo ""
  echo "Check logs in ${LOG_DIR}/ for details"
  echo ""
  echo "Note: Failures are expected if models haven't been trained yet."
fi

echo ""
echo "Results saved with naming pattern:"
echo "  test_results_{test_dataset}_using_{train_dataset}_{model}_{timestamp}.csv"
echo ""
