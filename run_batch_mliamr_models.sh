#!/bin/bash
# Example batch script for ML-iAMR models with sequence-encoded data
# This demonstrates how to run ML-iAMR models in parallel

set -e

echo "==================================================================="
echo "  ML-iAMR Batch Training Script"
echo "==================================================================="

# Configure parallel execution
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Define datasets (should contain sequence-encoded data)
DATASETS=(
    "BDQ-Only-SeqData"
    # Add more sequence datasets here
)

# Define ML-iAMR models
MODELS=(
    "SVC_MLiAMR"
    "RF_MLiAMR"
    "LR_MLiAMR"
    "CNN_1D_MLiAMR"
    # "CNN_2D_MLiAMR"  # Only for FCGR data
)

# Optional: Specify encoding explicitly (LE, OHE, FCGR)
# This is REQUIRED for ML-iAMR models
ENCODING="LE"  # Change to "OHE" or "FCGR" as needed

# Create logs directory
mkdir -p logs_mliamr

# Track jobs
declare -a PIDS
declare -a JOB_NAMES

job_count=0

echo ""
echo "Starting training jobs..."
echo "-------------------------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        job_count=$((job_count + 1))
        job_name="${dataset}_${model}"
        log_file="logs_mliamr/${job_name}.log"
        
        echo "[$job_count] Starting: $job_name"
        
        # Always use explicit encoding for ML-iAMR models
        python main_v2.py -s "$dataset" -m "$model" -r crossval -e "$ENCODING" > "$log_file" 2>&1 &
        
        pid=$!
        PIDS+=($pid)
        JOB_NAMES+=("$job_name")
        
        echo "    PID: $pid"
        echo "    Log: $log_file"
    done
done

echo "-------------------------------------------------------------------"
echo "Total jobs started: $job_count"
echo ""
echo "Waiting for all jobs to complete..."
echo "(You can monitor progress with: tail -f logs_mliamr/*.log)"
echo ""

# Wait for all jobs and track failures
failed_jobs=0
successful_jobs=0

for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    job_name=${JOB_NAMES[$i]}
    
    if wait $pid; then
        echo "✓ SUCCESS: $job_name (PID: $pid)"
        successful_jobs=$((successful_jobs + 1))
    else
        echo "✗ FAILED: $job_name (PID: $pid)"
        echo "   Check log: logs_mliamr/${job_name}.log"
        failed_jobs=$((failed_jobs + 1))
    fi
done

echo ""
echo "==================================================================="
echo "  Batch Training Complete"
echo "==================================================================="
echo "Total jobs: $job_count"
echo "Successful: $successful_jobs"
echo "Failed: $failed_jobs"
echo ""

if [ $failed_jobs -gt 0 ]; then
    echo "⚠ Some jobs failed. Check the logs in logs_mliamr/ directory."
    exit 1
else
    echo "✓ All jobs completed successfully!"
    echo ""
    echo "Results saved as: output_{dataset}_{model}_{timestamp}.csv"
    echo "Models saved in: saved_models/"
    echo "Hyperparameters saved in: model_hyperparams.tsv"
fi

echo "==================================================================="
