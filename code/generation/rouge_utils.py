from rouge_score import scoring
import collections
import six


def create_ngrams(tokens, n):
    """
    Creates ngrams from the given list of tokens.
    token list에서 ngram 생성
    Args:
        tokens: A list of tokens from which ngrams are created.
        n: Number of tokens to use, e.g. 2 for bigrams.
    Returns:
        A dictionary mapping each bigram to the number of occurrences.
    """
    ngrams = collections.Counter()
    for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
        ngrams[ngram] += 1
    return ngrams


def score_ngrams(target_ngrams, prediction_ngrams):
    """Compute n-gram based rouge scores. (ROUGE-N)
    Args:
        target_ngrams: A Counter object mapping each ngram to number of occurrences for the target text.
        prediction_ngrams: A Counter object mapping each ngram to number of occurrences for the prediction text.
    Returns:
        A Score object containing computed scores.
    """
    intersection_ngrams_count = 0 
    
    # 중복되는 n-gram count
    for ngram in six.iterkeys(target_ngrams):
        intersection_ngrams_count += min(target_ngrams[ngram], prediction_ngrams[ngram])
        target_ngrams_count = sum(target_ngrams.values())
        prediction_ngrams_count = sum(prediction_ngrams.values())
        
        precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
        recall = intersection_ngrams_count / max(target_ngrams_count, 1)
        fmeasure = scoring.fmeasure(precision, recall)
    
    return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)