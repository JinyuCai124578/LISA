deepspeed --master_port=24999 train_lisaqwen_ds.py \
  --version="/mnt/shared-storage-user/caijinyu/model/models--lmms-lab--llava-onevision-qwen2-7b-ov-chat/snapshots/3e979daa252a57fe1fdc4b0f537bf03d9d062031" \
  --dataset_dir='/mnt/shared-storage-user/caijinyu/data' \
  --log_base_dir='/mnt/shared-storage-user/ai4sdata2-share/caijinyu/runs' \
  --vision-tower='/mnt/shared-storage-user/caijinyu/model/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3' \
  --vision_pretrained="/mnt/shared-storage-user/caijinyu/model/sam_vit_h_4b8939.pth" \
  --dataset="reason_seg_em" \
  --reason_seg_data="organelle||plantorgan||cremi||material" \
  --sample_rates="1" \
  --batch_size="4" \
  --exp_name="lisaqwen-em-qa-fullft-lr1e-5" \
  --val_dataset="reason_seg_em" \
  --steps_per_epoch="64" \
  --epochs="50"  \
  --lr="1e-5" \
  --use_gpt_qa \
  --train_from_scratch \
  --full_finetune


# sbatch -N 1 -n 6 --gres=gpu:1 -p vip_gpu_ailab -A ai4neuro /home/bingxing2/ailab/caijinyu/LISA/train_lisaqwen.sh