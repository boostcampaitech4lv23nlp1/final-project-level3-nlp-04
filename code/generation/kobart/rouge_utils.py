from rouge_score import scoring
import collections
import six


def get_lcs_table(ref, can):
    """
    Create 2-d LCS (Longest Common Subsequence) score table.
    """
    rows = len(ref)
    cols = len(can)
    lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref[i - 1] == can[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    return lcs_table


def score_lcs(target_tokens, prediction_tokens):
    """
    Computes LCS (Longest Common Subsequence) rouge scores.
    최장 길이로 매칭되는 문자열 측정
    Args:
        target_tokens: Tokens from the target text.
        prediction_tokens: Tokens from the predicted text.
    Returns:
        A Score object containing computed scores.
    """
    
    if not target_tokens or not prediction_tokens:
        return scoring.Score(precision=0, recall=0, fmeasure=0)

    # Compute length of LCS from the bottom up in a table (DP appproach).
    lcs_table = get_lcs_table(target_tokens, prediction_tokens)
    lcs_length = lcs_table[-1][-1]
    
    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(target_tokens)
    fmeasure = scoring.fmeasure(precision, recall)
    
    return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)

    
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