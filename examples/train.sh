#!/bin/sh
# set -x  

# RECIPE_DIR=$(dirname $(dirname $(realpath "$0")))/"models"/"configs"
# SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"



# # Arguments
# MODEL_NAME="Llama-3.2-1B-Instruct"
# TASK="sft"
# ACCELERATE_STRATEGY="deepspeed_zero3"
# NUM_PROCESSES=1

# # Parse command-line options
# while getopts m:t:p: flag
# do
#     case "${flag}" in
#         e) EXP_NAME=${OPTARG};;
#         m) MODEL_NAME=${OPTARG};;
#         t) TASK=${OPTARG};;
#         a) ACCELERATE_STRATEGY=${OPTARG};;
#         p) NUM_PROCESSES=${OPTARG};;
#         *) echo "Invalid option";;
#     esac
# done

# # Run the script
# ACCELERATE_LOG_LEVEL=info accelerate launch \
#     --config_file ${RECIPE_DIR}/accelerate_configs/${ACCELERATE_STRATEGY}.yaml \
#     --num_processes=${NUM_PROCESSES} \
#     ${SCRIPT_DIR}/finetune_v1.py ${RECIPE_DIR}/${MODEL_NAME}/config_full.yaml --exp_name ${EXP_NAME}

RECIPE_DIR=$(dirname $(dirname $(realpath "$0")))/"models"/"configs"
PRUNED_DIR=$(dirname $(dirname $(realpath "$0")))/"models"/"pruned"
FINETUNED_DIR=$(dirname $(dirname $(realpath "$0")))/"models"/"finetuned"
SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"
PROJECT_ROOT=$(dirname $(dirname $(realpath "$0")))

# Environment Setting
set -x  # print the commands

# Add project root to PYTHONPATH for both the current directory and LLMPruner
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Arguments
MODEL_NAME="Llama-3.2-1B-Instruct"
NUM_PROCESSES=1

# Parse command-line options
while getopts e:m:p:d:s:b: flag
do
    case "${flag}" in
        e) EXP_NAME=${OPTARG};;
        m) MODEL_NAME=${OPTARG};;
        p) NUM_PROCESSES=${OPTARG};;
        d) DATA_PATH=${OPTARG};;
        s) SPARSITY=${OPTARG};;
        b) BATCH_SIZE=${OPTARG};;
        *) echo "Invalid option";;
    esac
done
if [ "$SPARSITY" = "0.00" ]; then
    python "${SCRIPT_DIR}/finetune_v1.py" \
        "${RECIPE_DIR}/${MODEL_NAME}/config_full.yaml" \
        --exp_name="${EXP_NAME}" \
        --data_path="${DATA_PATH}" \
        --output_dir="${FINETUNED_DIR}/${EXP_NAME}" \
        --per_device_train_batch_size=${BATCH_SIZE} \
        --per_device_eval_batch_size=${BATCH_SIZE}
else
    python "${SCRIPT_DIR}/finetune_v1.py" \
        "${RECIPE_DIR}/${MODEL_NAME}/config_full.yaml" \
        --exp_name="${EXP_NAME}" \
        --data_path="${DATA_PATH}" \
        --model_name_or_path="${PRUNED_DIR}/${EXP_NAME}/pytorch_model.bin" \
        --output_dir="${FINETUNED_DIR}/${EXP_NAME}" \
        --per_device_train_batch_size=${BATCH_SIZE} \
        --per_device_eval_batch_size=${BATCH_SIZE}
fi