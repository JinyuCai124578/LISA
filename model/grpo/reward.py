import torch

def format_reward_cal(completions, **kwargs):
   """
   Assigns a reward for adhering to the desired XML format.
   whether [SEG] is in the completions
   """
   responses = [completion[0]['content'] for completion in completions]
   rewards = []
   format_scores = []
   for response in responses:
       score = 0.0
       if "[SEG]" in response: score += 0.2
       rewards.append(score)
       format_scores.append(score)
   return rewards

def correctness_reward_cal(completions, answers, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.
    answer class in completions: 1.0
    otherwise: 0.0
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for r, a in zip(responses, answers):
        a.replace("_", " ")
        if a.lower() in r.lower():  # Exact match case
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def iou_reward_cal(predictions, ground_truths, **kwargs):
    """
    Assigns a reward based on the Intersection over Union (IOU) metric.
    IOU = intersection / union
    """
    rewards = []
    for pred, gt in zip(predictions, ground_truths):
        intersection=torch.logical_and(pred, gt).sum()
        union=torch.logical_or(pred, gt).sum()
        iou=intersection/union
        rewards.append(iou)
    
    return rewards

def combined_reward(completions, answers, pred_masks, gt_masks, weight=(1,1,10), **kwargs):
    format_score= format_reward_cal(completions=completions)
    correctness_score= correctness_reward_cal(completions=completions, answers=answers)
    iou_score= iou_reward_cal(predictions=pred_masks, ground_truths=gt_masks)
    combined_rewards=[]
    for format_reward, correctness_reward, iou_reward in zip(format_score, correctness_score, iou_score):
        combined_reward= weight[0]*format_reward + weight[1]*correctness_reward + weight[2]*iou_reward
        combined_rewards.append(combined_reward)
    return combined_rewards