import numpy as np
import torch
import transformers

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)


version="xinlai/LISA-13B-llama2-v1"
tokenizer = transformers.AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        force_download=False,
    )

# Load model directly
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14")

seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

model_args = {
    "train_mask_decoder": True,
    "out_dim": 256,
    "ce_loss_weight": 1,
    "dice_loss_weight": 0.5,
    "bce_loss_weight": 2,
    "seg_token_idx": seg_token_idx,
    "vision_pretrained": '/mnt/shared-storage-user/caijinyu/model/sam_vit_h_4b8939.pth',
    "use_mm_start_end": True,
}

model = LISAForCausalLM.from_pretrained(
    version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,force_download=False, **model_args
)

