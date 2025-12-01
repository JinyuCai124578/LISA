
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2Config,
                          Qwen2ForCausalLM, Qwen2Model)
from model.llava.model.language_model.llava_qwen import LlavaQwen2Config,LlavaQwen2ForCausalLM

qwen_model_name="Qwen/Qwen2.5-14B"
qwen_model=Qwen2ForCausalLM.from_pretrained(qwen_model_name)
tokenizer=AutoTokenizer.from_pretrained(qwen_model_name)

# Step 2: 构建 LLaVA-Qwen 模型配置
config = LlavaQwen2Config.from_pretrained(qwen_model_name)
# 如果有自定义 config 设置，可以在这里修改
config.mm_vision_tower = "openai/clip-vit-large-patch14"  # 示例视觉模型

# Step 3: 初始化 LLaVA-Qwen 模型
llava_model = LlavaQwen2ForCausalLM(config)

# Step 4: 将 Qwen 权重复制到 LLaVA 模型中
llava_model.model.load_state_dict(qwen_model.model.state_dict(), strict=False)
llava_model.lm_head.load_state_dict(qwen_model.lm_head.state_dict(), strict=False)