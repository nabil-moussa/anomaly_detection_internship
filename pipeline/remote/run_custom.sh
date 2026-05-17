#!/bin/bash
#SBATCH --job-name=mtad_smap
#SBATCH --output=/Work/Users/nmoussa/mtad-gat-pytorch/logs/smap_%j.out
#SBATCH --error=/Work/Users/nmoussa/mtad-gat-pytorch/logs/smap_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu

module load anaconda3@2022.10/gcc-12.1.0
source activate mtad
export MPLCONFIGDIR=$WORK/matplotlib_cache
export LD_PRELOAD=/Home/Users/nmoussa/.conda/envs/mtad/lib/libstdc++.so.6
mkdir -p $MPLCONFIGDIR
mkdir -p /Work/Users/nmoussa/handoff   
cd /Work/Users/nmoussa/mtad-gat-pytorch

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python train.py \
  --dataset CUSTOM \
  --epochs 50 \
  --lookback 100 \
  --normalize False \
  --use_vae False \
  --use_gatv2 False \
  --gru_hid_dim 300 \
  --fc_hid_dim 300 \
  --recon_hid_dim 300 \
  --gamma 0.8 \
  --init_lr 0.001 \
  --bs 256 \
  --use_cuda True \
  --use_sr_cleaning False \
  --n_seeds 1 \
  --seeds 42
  
  