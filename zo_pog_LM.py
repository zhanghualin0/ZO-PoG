# import argparse
import logging
import torch
import math
import os
os.environ['NUMEXPR_MAX_THREADS'] = '128'
import random
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from torch.optim import Adam, AdamW, SGD
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    set_seed,
)
from transformers.utils.versions import require_version
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaForMaskedLM 
from torch.nn import CrossEntropyLoss
from loss import *
import wandb
from opt import OPT
from common import task_to_keys, LABEL2ID_CONFIG, LABEL_CONVERT, TEMPLATE_CONFIG, counter, ApiCallLimitError, Imbalanced_Metric, pmi, constrainScoreByWholeExact
logger = logging.getLogger(__name__)
from zo_pog_common import *
from torch.nn import functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--task_name", type=str, default=None, help="The name of the glue task.", choices=list(task_to_keys.keys()))
    parser.add_argument("--file_name", type=str, default=None, help="The name of the domain-specific task.")
    parser.add_argument("--n_prompt_tokens", default=50, type=int)
    parser.add_argument("--loss_type", default='ce', type=str)
    parser.add_argument("--model_name_or_path", default='roberta-large', choices=['roberta-large',], type=str)
    parser.add_argument("--prompt_learning_rate", type=float, default=1e-3)
    parser.add_argument("--p0_learning_rate", type=float, default=None)
    parser.add_argument("--sample_size", type=int, default=20, help="IMPORTANT, sample size per batch")
    parser.add_argument("--prompt_search_space", type=int, default=100)
    #parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Whether to run wandb.")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=450, help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."))
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--k_shot", default=16, type=int, help="-1 denotes full-shot")
    parser.add_argument("--api_limit", type=int, default=1000, help="The limit of the API request")
    parser.add_argument("--intrinsic_dim", default=500, type=int)
    #parser.add_argument("--popsize", default=20, type=int)
    #parser.add_argument("--bound", default=0, type=int)
    parser.add_argument("--sigma", default=1, type=float)
    parser.add_argument("--alpha", default=100.0, type=float)
    parser.add_argument("--print_every", default=50, type=int)
    parser.add_argument("--eval_every", default=100, type=int)
    #parser.add_argument("--alg", default='CMA', type=str)
    parser.add_argument("--random_proj", default='normal', type=str)
    parser.add_argument("--cat_or_add", default='add', type=str)
    parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')
    parser.add_argument("--inference_framework", default='pt', type=str, help='''Which inference framework to use. Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively''')
    parser.add_argument("--onnx_model_path", default=None, type=str, help='Path to your onnx model.')
    
    parser.add_argument("--muz", default=0.01, type=float)
    parser.add_argument("--use_ngram", action="store_false", help="If True, will extract ngrams and use them.")
    parser.add_argument("--tau", default=1.0, type=float)

    args = parser.parse_args()
    #args.popsize = args.popsize if args.popsize > 0 else  4 + 3 * np.log(args.intrinsic_dim)
    # alg = args.alg
    # random_proj = args.random_proj
    # print_every = args.print_every
    # eval_every = args.eval_every
    # cat_or_add = args.cat_or_add

    if args.inference_framework not in ['pt', 'ort']:
        raise ValueError(f'inference_framework only supports "pt", "ort", got `{args.inference_framework}` instead.')
    if args.inference_framework == 'ort':
        assert args.onnx_model_path is not None, 'Path to onnx model is required, got None instead.'
        assert os.path.exists(args.onnx_model_path), f'In valid onnx model path `{args.onnx_model_path}`'

    args.init_prompt_path = None

    args.train_file = './dataset/' + args.file_name + '/train.csv' if args.file_name else None
    args.validation_file = './dataset/' + args.file_name + '/dev.csv' if args.file_name else None
    args.test_file = './dataset/' + args.file_name + '/test.csv' if args.file_name else None
    if args.task_name in ["book", "elec"]:
        args.train_file = './dataset/' + args.task_name + '/train.csv' if args.task_name else None
        args.validation_file = './dataset/' + args.task_name + '/dev.csv' if args.task_name else None
        args.test_file = './dataset/' + args.task_name + '/test.csv' if args.task_name else None
    args.ngram_list = pmi(args)

    sanity = not (args.task_name and args.file_name)
    assert sanity
    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    return args

def main():
    args = parse_args()
    assert args.task_name != 'stsb'

    # specify a unique experiment_id for load_metric() otherwise will cause ERROR when having multiple run on a same server!
    task_name = args.task_name if args.task_name else args.train_file
    args.unique_task_name = task_name.replace("/", ".")
    args.experiment_id = 'zp_bbt_LM' + task_name + str(args.n_prompt_tokens) + str(args.prompt_learning_rate) + str(args.seed) + str(args.prompt_search_space) + args.loss_type

    if args.use_wandb:
        args.group_name = f"{args.model_name_or_path}_BDPL_" + task_name
        wandb.init(config=args, project="blackbox_prompt", group=args.group_name)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # download the dataset.
    if args.task_name is not None:
        if args.task_name in ["book", "elec"]:
            data_files = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
            if args.test_file is not None:
                data_files["test"] = args.test_file
            extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
            raw_datasets = load_dataset(extension, data_files=data_files)
        elif args.task_name in task_to_keys.keys():
            cache_dir = './download/Data/' + args.task_name
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            if args.task_name == 'snli':
                raw_datasets = load_dataset(args.task_name, 'plain_text', cache_dir=cache_dir)
                raw_datasets = raw_datasets.filter(lambda example: example['label'] in [0, 1, 2])
            elif args.task_name == 'agnews':
                raw_datasets = load_dataset('ag_news', 'default', cache_dir=cache_dir)
            else:
                raw_datasets = load_dataset('glue', args.task_name, cache_dir=cache_dir)
                #raw_datasets = load_dataset("glue", args.task_name)
        else:
            raise(NotImplementedError)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if args.task_name:
        label_to_id = LABEL2ID_CONFIG[args.task_name]
    elif args.file_name:
        label_to_id = LABEL2ID_CONFIG[args.file_name]

    num_labels = len(label_to_id)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model_cache_dir = './download/model/' + args.model_name_or_path
    if not os.path.exists(model_cache_dir):
        os.makedirs(model_cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, cache_dir=model_cache_dir)
    
    args.device = torch.device("cuda", args.cuda)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        if args.task_name is not None:
            template_cfg = TEMPLATE_CONFIG[args.task_name]
        elif args.file_name is not None:
            template_cfg = TEMPLATE_CONFIG[args.file_name]
        template_base = template_cfg.replace('[MASK]', tokenizer.mask_token)

        texts = []
        template = [template_base] * len(examples[sentence1_key])
        offset = 1000
        prompt = tokenizer.decode(list(range(offset, offset+args.n_prompt_tokens)))
        if sentence2_key:
            for tuple_ in list(zip(examples[sentence1_key], template, examples[sentence2_key])):
                sent_1 = tokenizer.tokenize(tuple_[0])[:200]
                new_sent_1 = tokenizer.convert_tokens_to_string(sent_1)
                sent_2 = tokenizer.tokenize(tuple_[2])[:200]
                new_sent_2 = tokenizer.convert_tokens_to_string(sent_2)
                texts.append(prompt + " . " + new_sent_1 + tokenizer.sep_token + new_sent_2 + tuple_[1])
                #texts.append(new_sent_1 + tokenizer.sep_token + new_sent_2 + tuple_[1])
        else:
            for tuple_ in list(zip(examples[sentence1_key], template)):
                sent_1 = tokenizer.tokenize(tuple_[0])[:400]
                new_sent_1 = tokenizer.convert_tokens_to_string(sent_1)
                texts.append(prompt + " . " + new_sent_1 + tuple_[1])
                #texts.append(new_sent_1 + tuple_[1])
        result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)

        if args.task_name:
            label_list = []
            for raw_label in examples["label"]:
                label = LABEL_CONVERT[args.task_name][raw_label]
                target_encodings = tokenizer.encode(str(label).lower(), add_special_tokens=False)
                label_list.append(target_encodings[0])
            result["labels"] = torch.tensor(label_list)
        elif args.file_name in DOMAIN_DATASET:
            label_list = []
            for raw_label in examples["label"]:
                label = LABEL_CONVERT[args.file_name][raw_label]
                target_encodings = tokenizer.encode(str(label).lower(), add_special_tokens=False)
                label_list.append(target_encodings[0])
            result["labels"] = torch.tensor(label_list)
        else:
            target_encodings = tokenizer.batch_encode_plus(examples["label"], add_special_tokens=False)
            result["labels"] = torch.tensor(target_encodings['input_ids']).squeeze(dim=1).to(args.device)
        result["mask_pos"] = torch.tensor([r.index(tokenizer.mask_token_id) for r in result['input_ids']]).to(args.device)   
        return result

    with accelerator.main_process_first():
        if args.k_shot >= 0:
            args.balance = True
            def preprocess_function_k_shot(examples):
                random_indices = list(range(0, len(examples["label"])))
                random.shuffle(random_indices)

                new_examples = {}
                for key in examples.keys():
                    new_examples[key] = []
                label_count = {}

                for index in random_indices:
                    label = examples['label'][index]
                    if label not in label_count:
                        label_count[label] = 0
                    if label_count[label] < args.k_shot:
                        for key in examples.keys():
                            new_examples[key].append(examples[key][index])
                        label_count[label] += 1
                
                print("Finish few-shot sampling!")

                result = preprocess_function(new_examples)
                return result
            # k-shot learning
            raw_train_dataset_split = raw_datasets["train"].train_test_split(test_size=0.5)
            raw_train_dataset = raw_train_dataset_split['train']
            raw_eval_dataset = raw_train_dataset_split['test']
            train_dataset = raw_train_dataset.map(
                preprocess_function_k_shot,
                batched=True,
                batch_size=1000000,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            eval_dataset = raw_eval_dataset.map(
                preprocess_function_k_shot,
                batched=True,
                batch_size=1000000,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            if args.task_name == 'mnli':
                test_dataset = raw_datasets["validation_matched"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
                test_dataset_mm = raw_datasets["validation_mismatched"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
            elif args.task_name == 'agnews':
                test_dataset = raw_datasets["test"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
            else:
                test_dataset = raw_datasets["validation"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
        print("length of train data", len(train_dataset))
        print("length of eval data", len(eval_dataset))
        print("length of test data", len(test_dataset))

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    if args.task_name == 'mnli' and args.k_shot >= 0:  
        test_dataloader_mm = DataLoader(test_dataset_mm, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        test_dataloader_mm = accelerator.prepare(test_dataloader_mm)
    else:
        test_dataloader_mm = None
    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(train_dataloader, eval_dataloader, test_dataloader)

    model_forward_api = LMForwardAPI(args,
        model_name=args.model_name_or_path,
        task_name=args.task_name,
        # save_path=save_path,
        loss_type=args.loss_type,
        init_prompt_path=args.init_prompt_path,
        test_dataloader=test_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer
    )

    # if args.loss_type == 'ce':
    #     loss_fn = model_forward_api.ce_loss
    # elif args.loss_type == 'hinge':
    #     loss_fn = model_forward_api.margin
    print(f"{args.loss_type}_loss")

    metric = Imbalanced_Metric(args.task_name, args.experiment_id)

    def run_zo(args, batch, z_t, prompts_probs):
        z_t_tmp = copy.deepcopy(z_t).repeat(len(batch['input_ids']), 1)

        prompts_dist = torch.distributions.Categorical(prompts_probs)
        prompts_discrete_indices_list = []
        g_t_hats = []
        for k in range(int(args.sample_size/2)):
            prompts_discrete_indices = prompts_dist.sample()
            prompts_discrete_indices_list.append(prompts_discrete_indices)
            indices_list = prompts_discrete_indices.int().tolist()
            new_batch = update_p0(args, batch, prompts_probs, indices_list)
            
            noise = torch.normal(mean=0.0, std=1.0, size=z_t_tmp.shape).to(args.device)

            z_t_tmp1 = z_t_tmp - args.muz * noise
            loss_before_noise, query_numbers = model_forward_api.my_eval(args.model_name_or_path, metric, z_t_tmp1, new_batch)
            z_t_tmp2 = z_t_tmp + args.muz * noise
            loss_after_noise, query_numbers = model_forward_api.my_eval(args.model_name_or_path, metric, z_t_tmp2, new_batch)

            delta_loss = (loss_after_noise - loss_before_noise).reshape(-1, 1).repeat((1,)+z_t_tmp.shape[1:])

            g_t_hat = torch.mean(delta_loss * noise / (2*args.muz), dim=0)

            g_t_hats.append(g_t_hat.unsqueeze(0))
        
        g_t_hat = torch.mean(torch.cat(g_t_hats, dim=0), dim=0)
        return g_t_hat, query_numbers

    def run_policy(args, batch, prompts_alpha, prompts_probs, z_t):
        prompts_dist = torch.distributions.Categorical(prompts_probs)
        prompts_discrete_indices_list = []
        loss_list = []
        for k in range(args.sample_size):
            prompts_discrete_indices = prompts_dist.sample()
            prompts_discrete_indices_list.append(prompts_discrete_indices)
            indices_list = prompts_discrete_indices.int().tolist()
            new_batch = update_p0(args, batch, prompts_probs, indices_list)
            loss, query_numbers = model_forward_api.my_eval(args.model_name_or_path, metric, z_t.repeat(len(batch['input_ids']), 1), new_batch)
            loss = torch.mean(loss)
            loss_list.append(loss.item())
        
        loss_avg = sum(loss_list) / args.sample_size

        derivative = (- prompts_probs / (args.tau * prompts_alpha)).repeat(args.sample_size, 1, 1)
        for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
            for i in range(args.n_prompt_tokens):
                derivative[k][i][prompts_discrete_indices[i]] = (1 - prompts_probs[i][prompts_discrete_indices[i]]) / (args.tau * prompts_alpha[i][prompts_discrete_indices[i]])

        g_t_hat = torch.zeros_like(prompts_probs)
        for k in range(args.sample_size):
            g_t_hat += 1 / (args.sample_size - 1) * (loss_list[k] - loss_avg) * derivative[k]

        return g_t_hat, query_numbers
    
    z_t = torch.zeros(args.intrinsic_dim).type(torch.float32).to(args.device)

    prompts_alpha = torch.FloatTensor([[1 / args.prompt_search_space] * args.prompt_search_space] * args.n_prompt_tokens)
    prompts_probs = F.gumbel_softmax(torch.log(prompts_alpha), tau=args.tau)

    try:
        while True:
            for step, batch in enumerate(train_dataloader):

                g_t_hat, query_numbers = run_policy(args, batch, prompts_alpha, prompts_probs, z_t)

                prompts_alpha = prompts_alpha - args.p0_learning_rate * g_t_hat

                prompts_alpha[prompts_alpha <= 1e-11] = 1e-11

                prompts_probs = F.gumbel_softmax(torch.log(prompts_alpha), tau=args.tau)

                g_t_hat, query_numbers = run_zo(args, batch, z_t, prompts_probs)
                
                z_t = z_t - args.prompt_learning_rate * g_t_hat

                #print(f'query_numbers={query_numbers}, args.api_limit={args.api_limit}')
                if query_numbers >= args.api_limit:
                    raise ApiCallLimitError() 
                
            model_forward_api.my_dev_eval(args.model_name_or_path, metric, z_t, prompts_probs, args)

    except ApiCallLimitError:
        pass

    if args.balance:
        folder = f'balance/{args.task_name}_kshot_{args.k_shot}/zp_bbt_{args.model_name_or_path}_Lavg'
    file_path = f'results/{args.loss_type}/{folder}/' + f'{args.api_limit}_{args.n_prompt_tokens}_{args.k_shot}_{args.prompt_learning_rate}_{args.p0_learning_rate}'
    file_path = file_path + f'_{args.loss_type}/seed_{args.seed}'
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, f'{args.task_name}_result.txt')
    #test_metric = model_forward_api.eval(args.model_name_or_path, metric, test_dataloader=test_dataloader)
    test_metric, test_logits, test_labels = model_forward_api.my_test_eval(args.model_name_or_path, metric, args)
    with open(file_path, 'w') as f:
       f.write(f"On Test: total_api_count {query_numbers}, test_metric_results {test_metric} \n")
       f.write(f"finished\n")

if __name__ == "__main__":
    main()
