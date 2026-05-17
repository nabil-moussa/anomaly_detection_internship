#!/bin/bash
#SBATCH --job-name=mtad_msl
#SBATCH --output=/Work/Users/nmoussa/mtad-gat-pytorch/logs/mtad_%j.out
#SBATCH --error=/Work/Users/nmoussa/mtad-gat-pytorch/logs/mtad_%j.err
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

cd /Work/Users/nmoussa/mtad-gat-pytorch

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

#python train.py \
#    --use_vae True\
#    --use_sr_cleaning True \
#    --dataset msl \
#    --epochs 100 \
#    --bs 256 \
#    --init_lr 1e-3 \
#    --lookback 100 \
#    --use_cuda True \
#    --val_split 0.1 \
#    --print_every 1 \
#    --log_tensorboard False

python train.py --dataset msl --use_vae True --use_gatv2 False --level 0.98 --q 0.001\
  --gru_hid_dim 300 --fc_hid_dim 300 --recon_hid_dim 300 \
  --gamma 0.8 --epochs 100 --init_lr 0.001 --use_sr_cleaning True\
  --n_seeds 1 --seeds 42 #,123,456,789,1234