from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    train_data: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use."},
    )

    eval_data: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use."},
    )

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    wandb_project: str = field(
        default="None_project",
    )

    wandb_entity: Optional[str] = field(
        default="sajo-tuna",
    )

    wandb_name: str = field(
        default="None_name",
    )


@dataclass
class ConfigArguments:
    """
    Arguments for config setting
    """

    min_target_length: int = field(
        default=1
    )

    max_target_length: int = field(
        default=128
    )

    no_repeat_ngram_size: int = field(
        default=2
    )

    early_stopping: bool = field(
        default=True
    )

    length_penalty: float = field(
        default=0.0
    )

    num_labels: int = field(
        default=1
    )

    num_beams: int = field(
        default=4
    )