deepspeed --master_port=24999 train_ds.py \
  --version="/mnt/shared-storage-user/caijinyu/model/models--liuhaotian--llava-llama-2-13b-chat-lightning-preview/snapshots/bcda0227a7f371117a195ef0af88c1616a830520" \
  --dataset_dir='/mnt/shared-storage-user/caijinyu/data' \
  --log_base_dir="/mnt/shared-storage-user/ai4sdata2-share/caijinyu/runs" \
  --vision_pretrained="/mnt/shared-storage-user/caijinyu/model/sam_vit_h_4b8939.pth" \
  --dataset="reason_seg_em" \
  --reason_seg_data="organelle||plantorgan||cremi||material" \
  --sample_rates="1" \
  --batch_size="8" \
  --exp_name="lisa-em-qa-fullft-lr1e-5-scratch" \
  --val_dataset="reason_seg_em" \
  --steps_per_epoch="32" \
  --epochs="50"  \
  --lr="1e-5" \
  --use_gpt_qa \
  --full_from_scratch \
  # --train_mask_decoder_only \
  # --full_from_scratch \
  