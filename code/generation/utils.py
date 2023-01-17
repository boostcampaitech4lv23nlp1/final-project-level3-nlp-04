import re
import six
import numpy as np
from konlpy.tag import Mecab
from rouge_score import scoring
from rouge_utils import *


class KoreanRougeScorer(scoring.BaseScorer):
    def __init__(self, rouge_types):
        self.rouge_types = rouge_types
        self.tokenizer = Mecab()
        
    def score(self, target, prediction):
        target_tokens = self.tokenizer.morphs(target)
        prediction_tokens = self.tokenizer.morphs(prediction)
        result = {}
        
        for rouge_type in self.rouge_types:
            # ROUGE-N
            if re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
                n = int(rouge_type[5:]) # ex) rouge1 => 1
                if n <= 0:
                    raise ValueError(f"rougen requires positive n: {rouge_type}")
                target_ngrams = create_ngrams(target_tokens, n)
                prediction_ngrams = create_ngrams(prediction_tokens, n)
                scores = score_ngrams(target_ngrams, prediction_ngrams)
            else:
                raise ValueError(f"Invalid rouge type: {rouge_type}")
            result[rouge_type] = scores
        
        return result


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def compute(predictions, references):
    # ROUGE-N(unigram, bigram), ROUGE-L 측정
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    scorer = KoreanRougeScorer(rouge_types=rouge_types)
    aggregator = scoring.BootstrapAggregator()
    
    for ref, pred in zip(references, predictions):
        aggregator.add_scores(scorer.score(ref, pred))
        
    result = aggregator.aggregate()
    return result


def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred
    
    # labels -100이면 교체
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # 텍스트로 디코딩
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    # ROUGE score 계산
    result = compute(predictions=decoded_preds, references=decoded_labels)
    
    # median scores 추출
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    
    return result