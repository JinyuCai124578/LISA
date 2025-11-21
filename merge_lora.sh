CUDA_VISIBLE_DEVICES="0" python merge_lora_weights_and_save_hf_model.py \
  --version="/mnt/shared-storage-user/caijinyu/model/models--liuhaotian--llava-llama-2-13b-chat-lightning-preview/snapshots/bcda0227a7f371117a195ef0af88c1616a830520"  \
  --weight="/mnt/shared-storage-user/ai4sdata2-share/caijinyu/runs/lisa-em-qa-ft-lr1e-5-segdecoder/pytorch_model.bin" \
  --save_path="/mnt/shared-storage-user/ai4sdata2-share/caijinyu/runs/lisa-em-qa-ft-lr1e-5-segdecoder/" \
  --train_mask_decoder_only \