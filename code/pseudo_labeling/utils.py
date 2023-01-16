import os
import logging
import sklearn
import numpy as np


from transformers.trainer_utils import get_last_checkpoint
from arguments import Arguments
from transformers import TrainingArguments
from typing import Any, Tuple
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

logger = logging.getLogger(__name__)

label2id = {'중립': 0, '행복': 1, '당황': 2,
            '분노': 3, '불안': 4, '슬픔': 5, '혐오': 6}

id2label = {idx: token for token, idx in label2id.items()}


def get_last_checkpoint(args: Arguments, training_args: TrainingArguments) -> Tuple[Any, int]:
    # last checkpoint 찾기.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    return last_checkpoint


def micro_f1(preds, labels):
    return f1_score(labels, preds, average="micro", labels=list(range(len(label2id)))) * 100.0


def auprc_score(probs, labels):
    labels = np.eye(len(label2id))[labels]

    score = np.zeros((len(label2id),))
    for c in range(len(label2id)):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)

    return np.average(score) * 100.0


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = micro_f1(preds, labels)
    auprc = auprc_score(probs, labels)
    acc = accuracy_score(labels, preds)

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc
    }
