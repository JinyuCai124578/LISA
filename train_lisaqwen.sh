#!/bin/bash
#SBATCH --gres=gpu:1
#加载环境，此处加载anaconda环境以及通过anaconda创建的名为pytorch的环境
module load compilers/cuda/12.2
module load cudnn/8.8.1.3_cuda12.x
source activate lisa
deepspeed --master_port=24999 train_lisaqwen_ds.py \
  --version="/home/bingxing2/ailab/group/ai4neuro/EM_segmentation/model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5" \
  --dataset_dir='/home/bingxing2/ailab/group/ai4neuro/EM_segmentation' \
  --vision_pretrained="/home/bingxing2/ailab/caijinyu/LISA/pretrained/sam_vit_h_4b8939.pth" \
  --dataset="reason_seg_em" \
  --reason_seg_data="organelle||plantorgan||cremi||material" \
  --sample_rates="1" \
  --batch_size="1" \
  --exp_name="lisa-em-qa-ft-500step-lr1e-5-segdecoder" \
  --val_dataset="reason_seg_em" \
  --steps_per_epoch="500" \
  --epochs="50"  \
  --lr="1e-5" \
  --use_gpt_qa \
  --train_from_scratch \
  --vlm_model='qwen' \
  --workers="0" \


# sbatch -N 1 -n 6 --gres=gpu:1 -p vip_gpu_ailab -A ai4neuro /home/bingxing2/ailab/caijinyu/LISA/train.sh
# sbatch -N 1 -n 6 --gres=gpu:1 -p vip_gpu_ailab_low -A ailab /home/bingxing2/ailab/caijinyu/LISA/train.sh