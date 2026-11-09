#!/usr/bin/env bash
# Replication runs of DLinear with the authors' code (CPU, MacBook Air, PyTorch 2.8.0, Python 3.9.25).
# Run from the repo root, after setup.sh, with the conda env active.
set -e
cd LTSF-Linear

# --- ETTh1, horizon 96 (paper: MSE 0.375 / MAE 0.399; ours: 0.384 / 0.405) ---
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv \
  --model_id ETTh1_336_96 --model DLinear --data ETTh1 --features M --seq_len 336 --pred_len 96 \
  --enc_in 7 --des Exp --itr 1 --batch_size 32 --learning_rate 0.005 --num_workers 0 --use_gpu False

# --- Weather, horizon 96 (paper: 0.176 / 0.237; ours: 0.178 / 0.244) ---
# TODO(ishraq): paste the exact command you ran (see `history | grep weather`); the authors' script is
#   scripts/EXP-LongForecasting/DLinear/weather.sh  -- append --num_workers 0 --use_gpu False

# --- ILI, horizon 24, look-back 104 (paper: 2.215 / 1.081; ours: 2.280 / 1.061) ---
# TODO(ishraq): paste the exact command you ran; the authors' script is
#   scripts/EXP-LongForecasting/DLinear/ili.sh  -- append --num_workers 0 --use_gpu False
