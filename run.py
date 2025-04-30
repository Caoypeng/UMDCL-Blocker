import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    default_data_collator,
)

from transformers.trainer_utils import is_main_process
import transformers

from arguments import DataArguments, ModelArguments, CLTrainArguments as TrainingArguments
from data import TrainDatasetForCL, CLCollatorWithPadding
from modeling import MPNetForCL
from trainer import CLTrainer

logger = logging.getLogger(__name__)

import time
def main():
    start_time = time.time()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

     # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
     # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    # loading config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("You have to specify either model_name_or_path or tokenizer_name")
    
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eod_id is not None:
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token_id = tokenizer.im_start_id
            tokenizer.eos_token_id = tokenizer.im_end_id

    if model_args.model_name_or_path:
        if "sbert" in model_args.model_name_or_path.lower():
            model = MPNetForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
            )
        else:
            raise NotImplementedError("The model is not supported")
    # model.resize_token_embeddings(len(tokenizer))

    # loading data
    datasets = TrainDatasetForCL(data_args, tokenizer)

    if not data_args.hard_negative:
        if training_args.do_train:
            train_dataset = datasets.datasets["train"].map(
                datasets.prepare_features,
                batched=True,
                num_proc=1,
                remove_columns=datasets.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )


    data_collator = default_data_collator if data_args.pad_to_max_length else CLCollatorWithPadding(tokenizer)
    if data_args.hard_negative:
        trainer = CLTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        trainer = CLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    trainer.model_args = model_args
    # print(model.loss_fct.lambda_)
    # print(model.loss_fct_sparse.lambda_)
    if training_args.do_train:
        # trainer.save_model()
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
    # print(model.loss_fct.lambda_)
    # print(model.loss_fct_sparse.lambda_)
    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time} 秒")
if __name__ == "__main__":
    main()