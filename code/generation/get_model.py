from transformers import AutoModelForSeq2SeqLM

def get_model_func(config, args, config_args, tokenizer):

    # config.eos_token_id = tokenizer.sep_token_id
    # config.pad_token_id = tokenizer.pad_token_id
    # config.forced_eos_token_id = tokenizer.eos_token_id
    config.min_length = config_args.min_target_length
    config.max_length = config_args.max_target_length
    config.temperature = config.temperature
    config.no_repeat_ngram_size = config_args.no_repeat_ngram_size
    config.early_stopping = config_args.early_stopping
    config.length_penalty = config_args.length_penalty
    config.num_labels = config_args.num_labels
    config.num_beams = config_args.num_beams 

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
    return model

