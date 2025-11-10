# launch_deepspeed.py
import sys
import subprocess

if __name__ == "__main__":
    command = [
        "deepspeed",
        "--master_port=24999",
        "train_ds.py",
        "--version=liuhaotian/llava-llama-2-13b-chat-lightning-preview",
        "--dataset_dir=/home/bingxing2/ailab/group/ai4neuro/EM_segmentation",
        "--vision_pretrained=/home/bingxing2/ailab/caijinyu/LISA/pretrained/sam_vit_h_4b8939.pth",
        "--dataset=reason_seg_em",
        "--sample_rates=1",
        "--exp_name=lisa-em-ft"
    ]

    subprocess.run(command)