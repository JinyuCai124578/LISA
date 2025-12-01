from pycocoevalcap.cider.cider import Cider
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(candidate, references):
    """
    计算BLEU分数
    
    参数:
        candidate: str - 生成的文本
        references: list[str] - 参考文本列表
        
    返回:
        float: BLEU分数 (0-1)
    """
    # 分词
    candidate_tokens = candidate.split()
    reference_tokens = [ref.split() for ref in references]
    
    # 使用平滑函数处理短句子
    smoothing = SmoothingFunction().method1
    
    # 计算BLEU-4
    score = sentence_bleu(reference_tokens, candidate_tokens, 
                         smoothing_function=smoothing)
    return score
    
def calculate_cider(candidates, references):
    """
    计算CIDEr分数
    
    参数:
        candidates: dict - {id: 生成的文本}
        references: dict - {id: [参考文本列表]}
        
    返回:
        float: CIDEr分数
    """
    scorer = Cider()
    score, scores= scorer.compute_score(references, candidates)
    return score, scores

def calculate_bertscore(candidates, references, lang="en"):
    """
    计算BERTScore
    
    参数:
        candidates: list[str] - 生成的文本列表
        references: list[str] - 参考文本列表
        lang: str - 语言代码
        
    返回:
        tuple: (精确率, 召回率, F1)
    """
    P, R, F1 = bert_score(candidates, references, lang=lang , 
                            model_type="/mnt/shared-storage-user/caijinyu/model/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59/",
                            num_layers=17)
    return P.mean().item(), R.mean().item(), F1.mean().item()

scorer = Cider()
candidate_id='1'
# references=['The organelles located along the cell periphery with elongated, oval shapes and smooth contours are chloroplasts. Their positioning adjacent to the cell walls maximizes light absorption, which is crucial for photosynthesis. The internal thylakoid membranes visible in these structures are essential for light-dependent reactions, confirming their identity as chloroplasts.']
# candidate='The organelles in question are chloroplasts. They are positioned along the periphery of the cell, exhibiting elongated, ovoid shapes with smooth contours. This arrangement is typical for maximizing light absorption, a key function of chloroplasts in photosynthesis.As a result, the segmented regions are marked as.'
references=['The nucleus is the organelle adjacent to the cell wall, exhibiting a uniform texture with slightly darker electron density due to the presence of chromatin. This characteristic helps in identifying the nucleus, which plays a central role in cellular organization and gene expression.']
candidate='The organelle adjacent to the cell wall with a uniform texture and slightly darker electron density is the vacuole. Its position and texture are consistent with its role in storing nutrients and waste products, as well as maintaining turgor pressure.'
cider_score, cider_scores = calculate_cider({candidate_id: [candidate]}, {candidate_id: references})
p,r,f1=calculate_bertscore([candidate], references)
bleu=calculate_bleu(candidate, references)
print(cider_score, cider_scores)
print(p,r,f1)
print(bleu)