import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Whether to run wandb.")
    parser.add_argument("--task_name", default='wnut_10', type=str, #required=True,
                        help="The name of the task  "
                             "'ontonote', 'ontonote4'] ")
    parser.add_argument("--data_dir", default='datasets/wnut', type=str, #required=True,
                    help="The input data dir,", choices=['datasets/cluener', 'datasets/cner',
                                                         'datasets/conll_03_english',
                                                         'datasets/ontonote',  'datasets/ontonote4',
                                                         'datasets/movie',
                                                         'datasets/restaurant'] )
    parser.add_argument("--model_type", default='bert', type=str, #required=True,
                        help="Model type selected ",
                        choices=['bert', 'albert', 'bare_gpt2', 'gpt2', 'generate',
                                 'chinese_pretrained_gpt2', 'bare_chinese_gpt2',
                                 'generate_label_embedding','chinese_generate',
                                 'label_embedding',  'filling_entity'])

    parser.add_argument("--train_limit", default=10000, type=int,
                        help="the total lines load from train.text(notice not the number of examples)")
    parser.add_argument("--eval_limit", default=100, type=int,
                        help="the total lines load from dev.text(notice not the number of examples)")
    parser.add_argument("--test_limit", default=100, type=int,
                        help="the total lines load from test.text(notice not the number of examples)")

    parser.add_argument("--logging_steps", type=int, default=2,
                        help="Log every X updates steps.")
    parser.add_argument("--use_sweep", action="store_true", default=False,
                        help="Whether to run sweep .")
    parser.add_argument("--use_random", action="store_true", default=False,
                        help="Whether to randomly add ** around half of the entities in the trian dataset .")
    parser.add_argument("--duplicate_train_data", action="store_true", default=False,
                        help="Whether to duplicate the train data and add ** around all the entities in the trian dataset .")

    parser.add_argument("--output_dir", default='outputs/gpt2', type=str, #required=True,
                        help="The output directory where "
                             "the model predictions and checkpoints will be written."
                             " In my implementation, I mkdir the files listed in choices, you can mkdir your own output file",
                        choices=['outputs/cluener_output/gpt2', 'outputs/conll2003_output/gpt2',
                                 'outputs/cner_output/gpt2','outputs/conll2003_output/bert',
                                 'outputs/ontonote_output/gpt2', 'outputs/ontonote4_output/gpt2', 'outputs/ontonote4_output/bert'] )
    parser.add_argument("--output_file_dir", default='output_files/conll2003_output/', type=str, #required=True,
                        help="The output directory where the model predictions and checkpoints will be written,",
                        choices=['output_files/cluener_output/', 'output_files/conll2003_output/',
                                 'output_files/cner_output/', 'output_files/ontonote_output/', 'output_files/ontonote4_output/',
                                 'output_files/ontonote4_output/bert', ])

    parser.add_argument("--note", default='', type=str,
                        help="the implementation details to remind")
    parser.add_argument("--save_model", default=False, action="store_true",
                        help="Whether to save the model checkpoints, currently, there is no need to save the checkpoints.")
    parser.add_argument("--model_name_or_path", default='bert-base-cased',
                        type=str, #required=True,
                        help="Path to pre-trained model or shortcut name. ",
                        choices=['gpt2', 'gpt2-large','gpt2-medium', 'bert-base-chinese', 'bert-base-cased'])
    parser.add_argument("--output_file_name", default='test.json',
                        type=str, #required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument('--label_all_tokens', action="store_true", default=False,
                        help='whether to label all the tokens, otherwise will label split tokens(except for the first part in a word) with -100')

    # model_name_or_path can only be selected in the following list:
    # bart: 'facebook/bart-large'
    # 'bert-base-uncased'
    # 'bert-large-uncased'
    # 'bert-base-cased’
    # 'bert-large-cased'
    # 'bert-base-multilingual-uncased'
    # 'bert-base-multilingual-cased'
    # 'bert-base-chinese'
    # 'bert-base-german-cased'
    # 'bert-large-uncased-whole-word-masking'
    # 'bert-large-cased-whole-word-masking’
    # 'bert-large-uncased-whole-word-masking-finetuned-squad'
    # 'bert-large-cased-whole-word-masking-finetuned-squad'
    # 'bert-base-cased-finetuned-mrpc'
    # 'bert-base-german-dbmdz-cased'
    # 'bert-base-german-dbmdz-uncased'
    # "gpt2-medium"
    # "gpt2-large"
    # "distilgpt2"
    # "gpt2"
    # pretrained_model_name_or_path: either:
    # - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
    # - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
    # - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True
    # and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
    # a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
    # - None if you are both providing the configuration and state dictionary (resp. with keyword arguments ``config`` and ``state_dict``)

    parser.add_argument("--template", default='1', type=str, #required=True,
                        help="prompt size, you can modify the template in run_ner_xxx.py by changing TEMPLATE_CLASSES ",
                        choices=['-1', '0', '1', '2', '3', '4', 'a', 'b', 'c', 'd'])
    parser.add_argument("--learning_rate", default=5e-5, type=float,#bert default = 5e-5
                        help="The initial learning rate for Adam.")
    parser.add_argument("--crf_learning_rate", default=5e-5, type=float,#bert default = 5e-5
                        help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,#bert default =  0.01
                        help="Weight decay if we apply some.")
    parser.add_argument("--tokenizer_name", default='bert-base-cased', type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    # config name 和 tokenizer name 若为空则默认与 model_name_or_path一致,
    # I set the tokenizer for chinese as bert-base-chinese in run_ner_xxx.py and cannot be modified by --tokenizer_name.

    # Other parameters: always use the default values and haven't changed yet.
    parser.add_argument('--markup', default='bio', type=str,
                        choices=['biso', 'bio', 'bieso'])
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=32, type=int,#default = 128,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=32, type=int,#default = 128,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )

    parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
    parser.add_argument("--do_train", action="store_true", default=True,
                        help="Whether to run training.")
    parser.add_argument("--evaluate_and_test_during_training", action="store_true", default=True,
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_eval_with_saved_model", action="store_true", default=False,
                        help="Whether to run eval on the dev set with saved model.")
    parser.add_argument("--do_predict_with_saved_model", action="store_true", default=False,
                        help="Whether to run predictions on the test set with saved model.")

    # do_predict 在output_dir中加载存储的checkpoints
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    # adversarial training
    parser.add_argument("--do_adv", action="store_true", default=False,
                        help="Whether to adversarial training.")
    parser.add_argument('--adv_epsilon', default=1.0, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true", #default = False,
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true", default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true", default = True,
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html", )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    return parser