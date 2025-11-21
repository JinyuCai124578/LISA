#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2Config,
                          Qwen2ForCausalLM, Qwen2Model)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel

from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# =================== 1. 定义 Qwen2 的配置 ===================
class LlavaQwen2Config(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    """
    用于feature extractor的llm model
    仅作为一个抽象的组合类，组合文本和图像在进入llm之前的特征处理过程
    其中LlavaMetaModel用于注入额外的图像分支处理逻辑到feature extract逻辑中
    """
    config_class = LlavaQwen2Config

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2Model, self).__init__(config)


class LlavaQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM): # 多模态大模型的class
    config_class = LlavaQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)

        self.model = LlavaQwen2Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model( # 调用Qwen2ForCausalLM的forward进行普通LLM的前向传播
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs

if "llava" in CONFIG_MAPPING._mapping:
    # 移除模型类型
    del CONFIG_MAPPING._mapping["llava"]
AutoConfig.register("llava_qwen", LlavaQwen2Config)
AutoModelForCausalLM.register(LlavaQwen2Config, LlavaQwen2ForCausalLM)


if __name__ == "__main__":
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