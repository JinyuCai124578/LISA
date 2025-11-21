#!/bin/bash
#SBATCH --gres=gpu:1
#加载环境，此处加载anaconda环境以及通过anaconda创建的名为pytorch的环境
sleep inf
# sbatch -N 1 -n 6 --gres=gpu:1 -p vip_gpu_ailab -A ai4neuro /home/bingxing2/ailab/caijinyu/LISA/sleep.sh