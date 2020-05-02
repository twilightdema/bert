#!/bin/bash
# This is a convenience script for evaluating BERT on the GLUE benchmark.
#

set -ex

GLUE_DATA_DIR=glue_data
OUTPUT_DIR_BASE="output_glue"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/output"

BERT_CONFIG_FILE="pretrained_models/uncased_L-12_H-768_A-12/bert_config.json"
VOCAB_PATH="pretrained_models/uncased_L-12_H-768_A-12/vocab.txt"
INIT_CHECKPOINT="pretrained_models/uncased_L-12_H-768_A-12/bert_model.ckpt"

function run_task() {
  COMMON_ARGS="--output_dir="${OUTPUT_DIR}/$1" --data_dir="${GLUE_DATA_DIR}/$1" --vocab_file="${VOCAB_PATH}" --do_lower_case --max_seq_length=128 --optimizer=adamw --task_name=$1 --warmup_step=$2 --learning_rate=$3 --train_step=$4 --save_checkpoints_steps=$5 --iterations_per_loop=$5 --train_batch_size=$6"
  python3 -m run_classifier \
      ${COMMON_ARGS} \
      --do_train \
      --nodo_eval \
      --nodo_predict \
      --init_checkpoint="${INIT_CHECKPOINT}" \
      --bert_config_file="${BERT_CONFIG_FILE}"
  python3 -m run_classifier \
      ${COMMON_ARGS} \
      --nodo_train \
      --do_eval \
      --do_predict \
      --bert_config_file="${BERT_CONFIG_FILE}"
}

#run_task SST-2 1256 1e-5 20935 2093 32
run_task MNLI 1000 3e-5 10000 1000 128
run_task CoLA 320 1e-5 5336 533 16
#run_task QNLI 1986 1e-5 33112 3311 32
#run_task QQP 1000 5e-5 14000 1400 128
#run_task RTE 200 3e-5 800 80 32
#run_task STS-B 214 2e-5 3598 359 16
run_task MRPC 200 2e-5 800 80 32
