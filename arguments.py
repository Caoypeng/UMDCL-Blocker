import os
from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import Optional, List



@dataclass
class ModelArguments:
    """
    模型参数
    """
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


    # uBCL arguments
    model_type: Optional[str] = field(
        default="mpnet",
        metadata={"help": "The type of model to use (mpnet, bert, ...)."}
    )
    pool_type: Optional[str] = field(
        default="cls",
        metadata={"help": "The type of pooling layer."}
    )
    use_sparse: bool = field(
        default=False,
        metadata={
            "help": "Using sparse vector."
        },
    )
    initial_lambda: Optional[float] = field(
        default=10.0,
        metadata={"help": "The initial lambda value for adaptive loss."}
    )
    lambda_shape: Optional[List[int]] = field(
        default_factory=lambda: [1],
        metadata={"help": "The shape of lambda value."}
    )
    loss_type: Optional[str] = field(
        default="adaptive",
        metadata={"help": "The type of loss function to use (adaptive, simcse, ...)."}
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "The temperature value for SimCSE loss."}
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )



@dataclass
class DataArguments:
    """
    数据参数
    """
    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # uBCL arguments
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    delimiter: Optional[str] = field(
        default="####",
        metadata={"help": "The delimeter for the text file."}
    )
    grouped_size: Optional[int] = field(
        default=64,
        metadata={"help": "The number of examples in a group."}
    )

    # hard negative sample
    hard_negative: bool = field(
        default=False,
        metadata={"help": "Hard negative sampling."}
    )

    # dataset Augmentation arguments
    delete_word: bool = field(
        default=False,
        metadata={"help": "Delete words according to the given probability."}
    )
    delete_word_probability: float = field(
        default=None,
        metadata={"help": "The probability of deleting a word."}
    )
    swap_word: bool = field(
        default=False,
        metadata={"help": "Swap words according to the given probability."}
    )
    swap_word_probability: float  = field(
        default=None,
        metadata={"help": "The probability of swaping a word."}
    )
    replace_word: bool = field(
        default=False,
        metadata={"help": "Replace words according to the given probability."}
    )
    hnsw_index: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the hnsw index file (used for replace_word)."}
    )
    word2vec_model: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the word2vec model file (used for replace_word)."}
    )
    replace_word_probability: float  = field(
        default=None,
        metadata={"help": "The probability of replacing a word."}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_file):
            raise FileNotFoundError(f"cannot find file: {self.train_file}, please set a true path")


@dataclass
class CLTrainArguments(TrainingArguments):
    """
    训练参数
    """
    # adversarial training arguments
    adv_training: bool = field(
        default=False,
        metadata={
            "help": "Whether to use adversarial training."
        }
    )
    adv_eps: float = field(
        default=0.1,
        metadata={
            "help": "Epsilon for adversarial training."
        }
    )
    # lambda training arguments
    lambda_learning_rate: float = field(
        default=0.1,
        metadata={
            "help": "The learning rate of lambda."
        }
    )