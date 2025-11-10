#!/bin/bash
#SBATCH --gres=gpu:1
#加载环境，此处加载anaconda环境以及通过anaconda创建的名为pytorch的环境
module load compilers/cuda/12.2
module load cudnn/8.8.1.3_cuda12.x
source activate lisa
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export TRANSFORMERS_CACHE=/home/bingxing2/ailab/group/ai4neuro/EM_segmentation/model/cache
sleep 500000
# sbatch -N 1 -n 6 --gres=gpu:1 -p vip_gpu_ailab -A ai4neuro /home/bingxing2/ailab/caijinyu/LISA/sleep.sh