CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="/home/bingxing2/ailab/group/ai4neuro/EM_segmentation/model/lisa/models--liuhaotian--llava-llama-2-13b-chat-lightning-preview/snapshots/bcda0227a7f371117a195ef0af88c1616a830520"  \
  --weight="/home/bingxing2/ailab/group/ai4neuro/EM_segmentation/runs/lisa-em-qa-ft-500step-lr1e-5-segdecoder/pytorch_model.bin" \
  --save_path="/home/bingxing2/ailab/group/ai4neuro/EM_segmentation/runs/lisa-em-qa-ft-500step-lr1e-5-segdecoder/" 