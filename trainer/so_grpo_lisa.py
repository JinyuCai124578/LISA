"""
LISA SO-GRPO Training Implementation
Refactored from data_parallel.py to match so_grpo_qwsa.py structure
"""

import os
import sys
import logging
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import copy

# Add project root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))



class LISAGRPOTrainer:
    """LISA-specific SO-GRPO trainer with segmentation masks"""

    def __init__(self, model, reward_model, tokenizer, processor, args):
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.args = args
        
        # Policy network parameters
        self.policy_model = model
        self.value_model = model
        
        # GRPO specific parameters
        self.gamma = getattr(args, 'gamma', 0.99)
        self.lambda_gae = getattr(args, 'lambda_gae', 0.95)
        self.beta = getattr(args, 'beta', 0.1)  # KL penalty coefficient
        self.epsilon = getattr(args, 'epsilon', 0.2)  # PPO clip range
        
        # Reward weights
        self.lambda_mask = getattr(args, 'lambda_mask', 0.7)  # Mask quality weight
        self.lambda_text = getattr(args, 'lambda_text', 0.3)  # Text quality weight
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: args.learning_rate / (1 + getattr(args, 'eta', 1e-4) * step)
        )
        
        # Training statistics
        self.global_step = 0
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.kl_divs = []
        self.mask_rewards = []
        self.text_rewards = []
        self.gradient_variances = []
    
    def train_step(self, batch: Dict) -> Dict:
        """
        Execute one step of LISA GRPO training
        
        Args:
            batch: Dictionary containing:
                - prompt_ids: [batch_size, seq_len]
                - attention_masks_prompts: [batch_size, seq_len]
                - images: [batch_size, 3, H, W]
                - images_clip: [batch_size, 3, 224, 224]
                - masks_list: List of GT masks
                - label_list: List of label shapes
                - resize_list: List of resize info
                - classes_list: List of class labels
                - questions_list: List of questions
        
        Returns:
            Dictionary containing various losses and metrics
        """
        # 1. Generate multiple completions with masks
        num_generations = self.args.num_generations
        with torch.no_grad():
            rollout_data = self.generate_rollout_data(batch, num_generations)
        
        # 2. Compute rewards (mask + text quality)
        rewards = self.compute_comprehensive_rewards(rollout_data)
        
        # 3. Compute advantages using GAE
        advantages, returns = self.compute_gae_advantages(rewards)
        
        # 4. Compute old policy probabilities (from rollout)
        old_log_probs = rollout_data["old_log_probs"]
        
        # 5. Re-compute current policy probabilities (with gradients)
        token_log_probs = self.compute_log_probs(
            rollout_data["input_ids"],
            rollout_data["attention_mask"],
            rollout_data["logits_to_keep"]
        )
        
        # 6. Compute GRPO loss
        policy_loss, kl_div = self.compute_grpo_loss(
            token_log_probs,
            old_log_probs,
            advantages,
            rollout_data["completion_mask"],
            rollout_data["ref_log_probs"]
        )
        
        # 7. Total loss
        total_loss = policy_loss
        
        # 8. Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Record gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), self.args.max_grad_norm
        )
        self.gradient_variances.append(grad_norm.item())
        
        self.optimizer.step()
        self.scheduler.step()
        
        # 9. Record statistics
        self.episode_rewards.extend(rewards.view(-1).detach().cpu().tolist())
        self.policy_losses.append(policy_loss.item())
        self.kl_divs.append(kl_div.item())
        self.mask_rewards.append(rollout_data.get("mask_reward", 0.0))
        self.text_rewards.append(rollout_data.get("text_reward", 0.0))
        
        self.global_step += 1
        
        return {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "total_loss": total_loss.item(),
            "avg_reward": rewards.mean().item(),
            "mask_reward": rollout_data.get("mask_reward", 0.0),
            "text_reward": rollout_data.get("text_reward", 0.0),
            "gradient_variance": grad_norm.item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
    
    def generate_rollout_data(self, batch: Dict, num_generations: int) -> Dict:
        """
        Generate rollout data with model completions
        
        Args:
            batch: Input batch from dataloader
            num_generations: Number of completions to generate per prompt
        
        Returns:
            Dictionary containing generation results and log probabilities
        """
        self.policy_model.eval()
        
        with torch.no_grad():
            # Extract batch components
            prompt_ids = batch["prompt_ids"]  # [batch_size, seq_len]
            prompt_mask = batch["attention_masks_prompts"]
            
            # Repeat for multiple generations
            prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
            prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
            
            # Process images and masks
            images, images_clip, gt_masks, label_list, resize_list = self._prepare_image_data(
                batch, num_generations
            )
            
            # Generate completions using model
            outputs = self.policy_model(
                images=images,
                images_clip=images_clip,
                input_ids=prompt_ids,
                attention_masks=prompt_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.args.max_completion_length,
                temperature=self.args.temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=True,
                early_stopping=False,
                grpo=True,
            )
            
            # Extract outputs
            output_ids = outputs['output_ids']
            prompt_length = prompt_ids.shape[1]
            completion_ids = output_ids[:, prompt_length:]
            completion_mask = self._create_completion_mask(
                completion_ids, self.tokenizer.eos_token_id
            )
            
            # Extract predicted masks
            pred_low_res_masks = outputs['pred_low_res_masks']
            pred_masks = self._postprocess_masks(
                pred_low_res_masks, resize_list, label_list
            )
            
            # Compute log probabilities
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            
            old_log_probs = self.compute_log_probs(input_ids, attention_mask, logits_to_keep)
            
            # Compute reference log probabilities
            ref_model = copy.deepcopy(self.policy_model)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
            
            ref_log_probs = self.compute_log_probs_ref(
                ref_model, input_ids, attention_mask, logits_to_keep
            )
            
            # Format completions for reward computation
            formatted_completions = [
                [{'content': self.tokenizer.decode(ids, skip_special_tokens=False)}]
                for ids in completion_ids
            ]
            
            # Get repeated answers
            answers = []
            for a in batch['classes_list']:
                answers += a
            repeated_answers = [a for a in answers for _ in range(num_generations)]
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "completion_mask": completion_mask,
                "old_log_probs": old_log_probs,
                "ref_log_probs": ref_log_probs,
                "logits_to_keep": logits_to_keep,
                "formatted_completions": formatted_completions,
                "repeated_answers": repeated_answers,
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "label_list": label_list,
                "resize_list": resize_list,
                "batch_size": len(batch['classes_list']),
                "num_generations": num_generations,
            }
    
    def _prepare_image_data(self, batch: Dict, num_generations: int) -> Tuple:
        """Prepare and repeat image data for multiple generations"""
        images_clip_list = []
        images_list = []
        gt_masks = []
        label_list = []
        resize_list = []
        
        for i in range(batch["images_clip"].size(0)):
            num_questions = len(batch["questions_list"][i])
            
            images_clip_list.append(
                batch["images_clip"][i:i+1].repeat_interleave(num_questions, dim=0)
            )
            images_list.append(
                batch["images"][i:i+1].repeat_interleave(num_questions, dim=0)
            )
            
            label_list += [batch["label_list"][i]] * num_questions
            resize_list += [batch["resize_list"][i]] * num_questions
        
        for masks in batch["masks_list"]:
            gt_masks += list(masks)
        
        # Concatenate and repeat for num_generations
        images_clip = torch.cat(images_clip_list, dim=0)
        images_clip = images_clip.repeat_interleave(num_generations, dim=0)
        
        images = torch.cat(images_list, dim=0)
        images = images.repeat_interleave(num_generations, dim=0)
        
        gt_masks_repeat = [gt_mask for gt_mask in gt_masks for _ in range(num_generations)]
        label_list_repeat = [label for label in label_list for _ in range(num_generations)]
        resize_list_repeat = [resize for resize in resize_list for _ in range(num_generations)]
        
        return images, images_clip, gt_masks_repeat, label_list_repeat, resize_list_repeat
    
    def _create_completion_mask(self, completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
        """Create mask for valid completion tokens (before EOS)"""
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device
        )
        mask_exists = is_eos.any(dim=1)
        eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
        
        sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(
            is_eos.size(0), -1
        )
        return (sequence_indices <= eos_idx.unsqueeze(1)).int()
    
    def _postprocess_masks(self, pred_low_res_masks: torch.Tensor, 
                          resize_list: List, label_list: List) -> List[torch.Tensor]:
        """Postprocess model-generated masks to original resolution"""
        pred_masks = []
        for i in range(len(pred_low_res_masks)):
            pred_low_res_mask = pred_low_res_masks[i:i+1]
            pred_mask = self.policy_model.module.model.visual_model.postprocess_masks(
                pred_low_res_mask,
                input_size=resize_list[i],
                original_size=label_list[i].shape
            )
            pred_masks.append((pred_mask[0][0] > 0).int())
        return pred_masks
    
    def compute_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                         logits_to_keep: int) -> torch.Tensor:
        """Compute log probabilities for current policy"""
        input_ids = torch.clamp(input_ids, min=0)
        logits = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            grpo=True
        ).logits[:, :-1, :]
        
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :]
        
        return self._selective_log_softmax(logits, input_ids)
    
    def compute_log_probs_ref(self, ref_model: torch.nn.Module, input_ids: torch.Tensor,
                             attention_mask: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
        """Compute log probabilities for reference policy"""
        with torch.no_grad():
            input_ids = torch.clamp(input_ids, min=0)
            logits = ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                grpo=True
            ).logits[:, :-1, :]
            
            input_ids = input_ids[:, -logits_to_keep:]
            logits = logits[:, -logits_to_keep:, :]
            
            return self._selective_log_softmax(logits, input_ids)
    
    @staticmethod
    def _selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute log softmax and gather probabilities for selected tokens"""
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    
    def compute_comprehensive_rewards(self, rollout_data: Dict) -> torch.Tensor:
        """
        Compute rewards combining mask and text quality
        
        Args:
            rollout_data: Dictionary containing generation results
        
        Returns:
            Tensor of shape [batch_size * num_generations] with rewards
        """
        pred_masks = rollout_data["pred_masks"]
        gt_masks = rollout_data["gt_masks"]
        formatted_completions = rollout_data["formatted_completions"]
        repeated_answers = rollout_data["repeated_answers"]
        num_generations = rollout_data["num_generations"]
        
        # Compute mask quality reward
        mask_rewards = self._compute_mask_rewards(pred_masks, gt_masks)
        
        # Compute text quality reward
        text_rewards = self._compute_text_rewards(formatted_completions, repeated_answers)
        
        # Store for logging
        rollout_data["mask_reward"] = mask_rewards.mean().item()
        rollout_data["text_reward"] = text_rewards.mean().item()
        
        # Combine rewards
        device = mask_rewards.device
        total_rewards = (
            self.lambda_mask * mask_rewards +
            self.lambda_text * text_rewards
        )
        
        return total_rewards
    
    def _compute_mask_rewards(self, pred_masks: List[torch.Tensor],
                             gt_masks: List[torch.Tensor]) -> torch.Tensor:
        """Compute mask segmentation quality rewards (Dice + IoU)"""
        rewards = []
        
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            if pred_mask is None:
                rewards.append(0.0)
                continue
            
            # Ensure same device
            pred_mask = pred_mask.to(gt_mask.device)
            
            # Compute Dice score
            pred_bool = pred_mask.bool()
            gt_bool = gt_mask.bool()
            
            intersection = (pred_bool & gt_bool).float().sum()
            pred_sum = pred_bool.float().sum()
            gt_sum = gt_bool.float().sum()
            
            dice = (2 * intersection + 1e-7) / (pred_sum + gt_sum + 1e-7)
            
            # Compute IoU
            union = (pred_bool | gt_bool).float().sum()
            iou = intersection / (union + 1e-7) if union > 0 else torch.tensor(0.0)
            
            # Combined score
            reward = 0.5 * dice + 0.5 * iou
            rewards.append(reward.item())
        
        return torch.tensor(rewards, dtype=torch.float32, device=gt_masks[0].device)
    
    def _compute_text_rewards(self, formatted_completions: List, repeated_answers: List) -> torch.Tensor:
        """Compute text quality rewards using reward model"""
        rewards = []
        
        for completion, answer in zip(formatted_completions, repeated_answers):
            try:
                # Use reward model to compute text quality
                text_reward = self.reward_model.compute_text_reward(
                    completion[0]['content'], answer
                )
                rewards.append(text_reward)
            except Exception as e:
                logging.warning(f"Text reward computation failed: {e}")
                rewards.append(0.0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.tensor(rewards, dtype=torch.float32, device=device)
    
    def compute_gae_advantages(self, rewards: torch.Tensor,
                              values: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: [batch_size * num_generations]
            values: Value function estimates (optional)
        
        Returns:
            Tuple of (advantages, returns)
        """
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(1)
        
        rewards_np = rewards.detach().cpu().numpy()
        
        if values is None:
            values_np = rewards_np
        else:
            values_np = values.detach().cpu().numpy()
        
        batch_size = rewards_np.shape[0]
        seq_len = rewards_np.shape[1] if len(rewards_np.shape) > 1 else 1
        
        if len(rewards_np.shape) == 1:
            rewards_np = rewards_np.reshape(-1, 1)
            values_np = values_np.reshape(-1, 1)
            seq_len = 1
        
        # Compute GAE
        advantages = np.zeros_like(rewards_np)
        returns = np.zeros_like(rewards_np)
        
        for b in range(batch_size):
            gae = 0
            for t in reversed(range(seq_len)):
                if t < seq_len - 1:
                    delta = rewards_np[b, t] + self.gamma * values_np[b, t + 1] - values_np[b, t]
                else:
                    delta = rewards_np[b, t] - values_np[b, t]
                
                gae = delta + self.gamma * self.lambda_gae * gae
                advantages[b, t] = gae
                
                if t < seq_len - 1:
                    returns[b, t] = rewards_np[b, t] + self.gamma * returns[b, t + 1]
                else:
                    returns[b, t] = rewards_np[b, t]
        
        # Standardize advantages
        advantages_flat = advantages.flatten()
        if advantages_flat.std() > 1e-8:
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        advantages = advantages_flat.reshape(advantages.shape)
        
        device = rewards.device
        return (
            torch.from_numpy(advantages).to(device),
            torch.from_numpy(returns).to(device)
        )
    
    def compute_grpo_loss(self, token_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                         advantages: torch.Tensor, completion_mask: torch.Tensor,
                         ref_log_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GRPO loss with PPO clipping and KL penalty
        
        Args:
            token_log_probs: Current policy log probs [seq_len, batch*gen]
            old_log_probs: Old policy log probs [seq_len, batch*gen]
            advantages: Advantage estimates [seq_len, batch*gen]
            completion_mask: Mask for valid tokens [seq_len, batch*gen]
            ref_log_probs: Reference policy log probs [seq_len, batch*gen]
        
        Returns:
            Tuple of (policy_loss, kl_div)
        """
        # Compute policy ratio
        ratio = torch.exp(token_log_probs - old_log_probs)
        
        # PPO clipping
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        surrogate_loss = torch.min(surr1, surr2)
        
        # KL penalty
        kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
        per_token_loss = surrogate_loss - self.beta * kl
        
        # Compute mean loss across valid tokens
        loss = -((per_token_loss * completion_mask).sum(dim=0) / completion_mask.sum(dim=0)).mean()
        kl_div = kl.mean()
        
        return loss, kl_div
    
    def get_training_statistics(self) -> Dict:
        """Get training statistics"""
        return {
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "std_reward": np.std(self.episode_rewards) if self.episode_rewards else 0,
            "avg_policy_loss": np.mean(self.policy_losses) if self.policy_losses else 0,
            "avg_kl_div": np.mean(self.kl_divs) if self.kl_divs else 0,
            "avg_mask_reward": np.mean(self.mask_rewards) if self.mask_rewards else 0,
            "avg_text_reward": np.mean(self.text_rewards) if self.text_rewards else 0,
            "gradient_variance": np.mean(self.gradient_variances) if self.gradient_variances else 0,
            "total_steps": self.global_step
        }