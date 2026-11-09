#!/usr/bin/env bash
# New-data run: Beijing Dongsi station with DLinear (run from the repo root after setup.sh).
set -e
bash beijing/download_beijing.sh
python beijing/prepare_beijing.py Dongsi
cd LTSF-Linear
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path beijing_Dongsi.csv \
  --model_id Beijing_Dongsi_336_96 --model DLinear --data custom --features M --target OT \
  --seq_len 336 --pred_len 96 --enc_in 11 --des Exp --itr 1 --batch_size 32 --learning_rate 0.005 \
  --num_workers 0 --use_gpu False
cd ..
python beijing/baselines_beijing.py
