import os
import numpy as np
import torch
import torch.nn.functional as F
import datasets
from datasets import Metric, MetricInfo, load_metric
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from datetime import datetime

DOMAIN_DATASET = ['CI', 'SE', 'RCT', 'HP']

task_to_keys = {
    "cola": ("sentence", None),
    "book": ("sentence", None),
    "elec": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "snli": ("premise", "hypothesis"),
    "agnews": ("text", None),
}

LABEL2ID_CONFIG = {
    "sst2": {" terrible": 0, " great": 1},
    "qqp": {" no": 0, " yes": 1},
    "mrpc": {" no": 0, " yes": 1},
    "cola": {" no": 0, " yes": 1},
    "wnli": {" no": 0, " yes": 1},
    "qnli": {" yes": 0, " no": 1},
    "rte": {" yes": 0, " no": 1},
    # "book": {" great": 0, " terrible": 1},
    # "elec": {" great": 0, " terrible": 1},
    "book": {" positive": 0, " negative": 1},
    "elec": {" positive": 0, " negative": 1},
    # "book": {" yes": 0, " no": 1},
    # "elec": {" yes": 0, " no": 1},
    "imdb": {" terrible": 0, " great": 1},
    "cr": {" terrible": 0, " great": 1},
    "mr": {" terrible": 0, " great": 1},
    "HP": {' unhelpful': 0, ' helpful': 1}, # review helpfulness
    "mpqa": {" terrible": 0, " great": 1},
    "mnli": {" yes": 0, " maybe": 1, " no": 2},
    "snli": {" yes": 0, " maybe": 1, " no": 2},
    "CI": {' background': 0, ' comparison': 1, ' extension': 2, ' future': 3, ' motivation': 4, ' use': 5},
    "SE": {' comparison': 0, ' conjunction': 1, ' evaluation': 2, ' feature': 3, ' hyponym': 4, ' part': 5, ' function': 6},
    "RCT": {' background': 0, ' conclusion': 1, ' method': 2, ' objective': 3, ' result': 4} ,
    "agnews": {' world': 0, ' sports': 1, ' business': 2, ' tech': 3},
}

LABEL_CONVERT = {
    "sst2": {0: ' terrible', 1: ' great'},
    "qqp": {0: ' no', 1: ' yes'},
    'mrpc': {0: ' no', 1: ' yes'},
    'cola': {0: ' no', 1: ' yes'},
    'wnli': {0:  ' no', 1: ' yes'},
    'qnli': {0: ' yes', 1: ' no'},
    'rte': {0: ' yes', 1: ' no'},
    "mnli": {0: ' yes', 1: ' maybe', 2: ' no'},
    # "book":{0: ' great', 1: ' terrible'},
    # "elec":{0: ' great', 1: ' terrible'},
    "book":{0: ' positive', 1: ' negative'},
    "elec":{0: ' positive', 1: ' negative'},
    # "book":{0: ' yes', 1: ' no'},
    # "elec":{0: ' yes', 1: ' no'},
    "snli": {0: ' yes', 1: ' maybe', 2: ' no'},
    'CI': {'Background': ' background', 'CompareOrContrast': ' comparison', 'Extends': ' extension', 'Future': ' future', 'Motivation': ' motivation', 'Uses': ' use'},
    'SE': {'COMPARE': ' comparison', 'CONJUNCTION': ' conjunction', 'EVALUATE-FOR': ' evaluation', 'FEATURE-OF': ' feature', 'HYPONYM-OF': ' hyponym', 'PART-OF': ' part', 'USED-FOR': ' function'},
    'RCT': {'BACKGROUND': ' background', 'CONCLUSIONS': ' conclusion', 'METHODS': ' method', 'OBJECTIVE': ' objective', 'RESULTS': ' result'},
    'HP': {False: ' unhelpful', True: ' helpful'},
    "agnews": {0: ' world', 1: ' sports', 2: ' business', 3: ' tech'},
}

TEMPLATE_CONFIG = {
    "mnli": " entailment? [MASK].",
    "qqp": "? [MASK],",
    "sst2": " It was [MASK].",
    "mrpc": "? [MASK],",
    "cola": " correct? [MASK].",
    "book": " It was [MASK].",
    "elec": " It was [MASK].",
    "wnli": " entailment? [MASK].",
    "qnli": " entailment? [MASK].",
    "rte": " entailment? [MASK].",
    "CI": " What is the intent? [MASK].", 
    "SE": " What is the relation? [MASK].",
    "RCT": " It is [MASK].",
    "HP": " It is [MASK].",
    "imdb": " It was [MASK].",
    "cr": " It was [MASK].",
    "snli": " entailment? [MASK].",
    "agnews": " It was [MASK].",
}

def pmi(args) -> list:
    if args.use_ngram:
        prefix = {"gpt2": "gpt2-xl_", "gpt2-xl": "gpt2-xl_", "roberta-large": "", "llama3": "llama3_"}
        flag_name = args.file_name if args.file_name else args.task_name
        result=[]
        with open(f"./pmi/{args.model_name_or_path}/{prefix[args.model_name_or_path]}" + flag_name.lower() + ".txt",'r') as f:
            for line in f:
                result = result + (list(line.strip('\n').split(',')))

        unique = []
        [unique.append(i) for i in result if not i in unique]
        ngram_index_list = list(map(int, unique))
        return ngram_index_list
    return []

def solve_v_total_exact(prompt_emb):
    k = 1
    a, b = -3, 0

    b = prompt_emb.max()
    def f(v):
        s = (prompt_emb - v).clamp(0, 1).sum()
        return s - k
    itr = 0

    v = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    return v, itr

def constrainScoreByWholeExact(prompt_embeds):
    for i in range(len(prompt_embeds)):
        v, itr = solve_v_total_exact(prompt_embeds[i])
        prompt_embeds[i].sub_(v).clamp_(0, 1)

def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count = wrapper.count + 1
        res = func(*args, **kwargs)
        if wrapper.count % 100 == 0:
            # print ("{0} has been used: {1}x".format(func.__name__, wrapper.count))
            pass
        return res
    wrapper.count = 0
    return wrapper

class ApiCallLimitError(Exception):
    pass

def train_acc(args, preds, converted_target, metric, accelerator):
    pred_label_ = preds.clone().detach()
    pred_label = pred_label_.argmax(dim=-1)
    predictions_pred = torch.nn.functional.softmax(pred_label_, dim=1)
    if args.task_name in {'mnli', 'snli'}:
        metric.add_batch(
            predictions=accelerator.gather(predictions_pred.cpu().numpy()),
            references=accelerator.gather(converted_target.cpu().numpy()),
        )
    else:
        metric.add_batch(
            predictions=accelerator.gather(pred_label.detach().cpu().numpy()),
            references=accelerator.gather(converted_target.cpu().numpy()),
        )
    return metric

def evaluate(args, model, eval_dataloader, metric, accelerator, epoch, api_count, prompts_probs=None, prompt_length=None, tokenizer=None, folder=None, M=None):
    prompts_discrete_indices = prompts_probs.argmax(1)

    if args.use_ngram:
        prompts_discrete_indices_ngram_list = []
        indices_list = prompts_discrete_indices.int().tolist()
        for idx in indices_list:
            prompts_discrete_indices_ngram_list.append(args.ngram_list[idx])
        prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)
    
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if args.trial and step >= 100:
                break
            bsz = len(batch['input_ids'])

            if args.use_ngram:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
            else:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)

            mask_pos=np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
            mask_pos = torch.tensor(mask_pos[-1])
            label_to_id = model.config.label2id 

            sequence_output = model(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"])
            # last_hidden_state = sequence_output[0].squeeze()
            last_hidden_state = sequence_output[0]
            logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

            label = batch["labels"].to(args.device)
            label_keys = list(label_to_id.keys())
            label_map = {}
            for target in label_keys:
                label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = label_to_id[target]
            converted_target = label.clone()
            for key, val in label_map.items():
                converted_target[label == key] = val
            if M == None:
                sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
                interest_index = [item[0] for item in sort_label_map]
                logits = logits[:, interest_index]
            else:
                logits = logits @ M
            pred_label = logits.argmax(dim=-1)
            # predictions = (torch.exp(logits)/torch.sum(torch.exp(logits), dim=1).unsqueeze(1))[:, 1]
            predictions_pred = torch.nn.functional.softmax(logits, dim=1)
            if args.task_name in {'mnli', 'snli'}:
                metric.add_batch(
                    predictions=accelerator.gather(predictions_pred.cpu().numpy()),
                    references=accelerator.gather(converted_target.cpu().numpy()),
                )
            else:
                metric.add_batch(
                    predictions=accelerator.gather(pred_label),
                    references=accelerator.gather(converted_target),
                )

    if args.file_name in DOMAIN_DATASET:
        eval_metric = metric.compute(average='macro')
    else:
        eval_metric = metric.compute()

    # print("** eval **")
    # print(f"epoch {epoch + 1}: {eval_metric}")
    # if args.balance == True:
    #     eval_result = eval_metric['acc']
    # else:
    #     eval_result = eval_metric['auc']
    eval_result = eval_metric['acc']
    if folder:
        write_results(args, folder, epoch, api_count, None, None, eval_metric, metric_state='eval')
    return eval_result

def test(args, model, test_dataloader, metric, accelerator, epoch, api_count, best_epoch, best_api_count, prompts_probs=None, prompt_length=None, tokenizer=None, test_dataloader_mm=None, folder=None, test_metric=None, M=None, tableName=None):
    if test_metric is not None:
        write_results(args, folder, epoch, api_count, best_epoch, best_api_count, test_metric, metric_state='test', tableName=tableName)
        return test_metric
    
    prompts_discrete_indices = prompts_probs.argmax(1)
    if args.use_ngram:
        prompts_discrete_indices_ngram_list = []
        indices_list = prompts_discrete_indices.int().tolist()
        for idx in indices_list:
            prompts_discrete_indices_ngram_list.append(args.ngram_list[idx])
        prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            if args.trial and step >= 100:
                break
            bsz = len(batch['input_ids'])
            
            if args.use_ngram:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
            else:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)

            mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id)
            mask_pos = torch.tensor(mask_pos[-1])
            label_to_id = model.config.label2id
            sequence_output = model(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"])
            # last_hidden_state = sequence_output[0].squeeze()
            last_hidden_state = sequence_output[0]
            logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

            label = batch["labels"].to(args.device)
            label_keys = list(label_to_id.keys())
            label_map = {}
            for target in label_keys:
                label_map[tokenizer.encode(target, add_special_tokens=False)[0]]  = label_to_id[target]
            converted_target = label.clone()
            for key, val in label_map.items():
                converted_target[label == key] = val
            if M == None:
                sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
                interest_index = [item[0] for item in sort_label_map]
                logits = logits[:, interest_index]
            else:
                logits = logits @ M
            pred_label = logits.argmax(dim=-1)
            # predictions = (torch.exp(logits)/torch.sum(torch.exp(logits), dim=1).unsqueeze(1))[:, 1]
            predictions_pred = torch.nn.functional.softmax(logits, dim=1)
            # print(pred_label, pred_label.shape)
            # print(converted_target, converted_target.shape)
            if args.task_name in {'mnli', 'snli'}:
                metric.add_batch(
                    predictions=accelerator.gather(predictions_pred.cpu().numpy()),
                    references=accelerator.gather(converted_target.cpu().numpy()),
                )
            else:
                metric.add_batch(
                    predictions=accelerator.gather(pred_label),
                    references=accelerator.gather(converted_target),
                )
            
    if args.file_name in DOMAIN_DATASET:
        test_metric = metric.compute(average='macro')
    else:
        test_metric = metric.compute()

    print("** test **")
    print(f"current epoch {epoch + 1}, best epoch {best_epoch + 1}, api_count {api_count}: {test_metric}")
    if args.use_wandb:
        for key in test_metric.keys():
            eval_key = 'Black_test_' + key
            wandb.log({eval_key: test_metric[key]})
    
    write_results(args, folder, epoch, api_count, best_epoch, best_api_count, test_metric, metric_state='test', tableName=tableName)
    return test_metric

def write_results(args, folder, epoch, api_count, best_epoch, best_api_count, metric, metric_state, tableName=None):
    assert metric_state in ['train', 'eval', 'test']
    args.file_path = f'ICLR_results/{folder}/' + f'{args.api_limit}_{args.prompt_length}_{args.prompt_search_space}_{args.per_device_train_batch_size}_{args.prompt_learning_rate}'
    if hasattr(args, 'M_learning_rate') and args.M_learning_rate:
        args.file_path += f'_{args.M_learning_rate}'
    prompt_optimizer_name = f"_{args.prompt_optimizer_name}" if args.prompt_optimizer_name else ""
    args.file_path += f'_{args.loss_type}{prompt_optimizer_name}/seed_{args.seed}'
    if not os.path.isdir(args.file_path):
        os.makedirs(args.file_path)
    args.file_path = os.path.join(args.file_path, f'{args.task_name}_result.txt')
    with open(args.file_path, 'a') as f:
        if metric_state == 'train' or metric_state == 'eval':
            f.write(f"current_epoch {epoch + 1}, total_api_count {api_count}, {metric_state}_metric_results {metric} \n")
        else:
            f.write(f"current_epoch {epoch + 1}, total_api_count {api_count}, best_eval_epoch {best_epoch + 1}, best_api_count {best_api_count}, {metric_state}_metric_results {metric} \n")
        if epoch + 1 == args.num_train_epochs and metric_state == 'test':
            f.write(f"finished\n")

def simple_accuracy(preds, labels):
    return float((np.array(preds) == np.array(labels)).mean())

class Imbalanced_Metric(Metric):
    def _info(self):
        if self.config_name not in [
            "sst2",
            "mnli",
            "mnli_mismatched",
            "mnli_matched",
            "cola",
            "stsb",
            "mrpc",
            "qqp",
            "qnli",
            "rte",
            "wnli",
            "hans",
            "book",
            "elec",
            "snli",
            "agnews",
        ]:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["sst2", "mnli", "mnli_mismatched", "mnli_matched", '
                '"cola", "stsb", "mrpc", "qqp", "qnli", "rte", "wnli", "hans", "snli"]'
            )
        return MetricInfo(
            description="",
            citation="",
            features=datasets.Features(
                {
                    "references": datasets.Value("int32"),
                    "predictions": datasets.Sequence(datasets.Value("float")),
                }
                if self.config_name in {"mnli", 'snli', 'agnews'}
                else {
                    "predictions": datasets.Value("float32" if self.config_name != "stsb" else "float32"),
                    "references": datasets.Value("int64" if self.config_name != "stsb" else "float32"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            # format="numpy",
        )
    
    def _compute(self, predictions, references):
        return self.all(predictions, references, self.config_name)
    
    @staticmethod
    def all(preds, labels, config_name=None):
        if config_name in {"mnli", 'snli', 'agnews'}:
            auc = roc_auc_score(y_true=labels, y_score=preds, multi_class='ovr')
            preds_label = np.argmax(preds, axis=1)
            acc = simple_accuracy(preds_label, labels)
            f1 = float(f1_score(y_true=labels, y_pred=preds_label, average='macro'))
            precision = float(precision_score(y_true=labels, y_pred=preds_label, average='macro'))
            recall = float(recall_score(y_true=labels, y_pred=preds_label, average='macro'))
            return {
                "acc": acc,
                "auc": auc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        else:
            auc = roc_auc_score(y_true=labels, y_score=preds, average="macro")
            #preds = np.array(np.array(preds)>0.5, dtype=int)
            acc = simple_accuracy(preds, labels)
            f1 = float(f1_score(y_true=labels, y_pred=preds))
            precision = float(precision_score(y_true=labels, y_pred=preds))
            recall = float(recall_score(y_true=labels, y_pred=preds))
            #! matthews_correlation: a measure of the quality of binary and multiclass classifications (see Matthews Correlation for more information). Its range of values is between -1 and +1, where a coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.
            m_corrcoef = matthews_corrcoef(y_true=labels, y_pred=preds)   
            return {
                "acc": acc,
                "auc": auc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "matthews_correlation": m_corrcoef
            }

def get_default_metric(args):
    if args.task_name in args.task_name in ["book","elec"]:
        metric = load_metric('accuracy', args.experiment_id)
    elif args.task_name is not None:
        metric = load_metric("glue", args.task_name, experiment_id=args.experiment_id)
    elif args.file_name in DOMAIN_DATASET:
        metric = load_metric('f1', args.experiment_id)
    return metric

def blppg_evaluate(args, model, eval_dataloader, metric, accelerator, epoch, api_count, prompts_probs=None, prompt_length=None, tokenizer=None, folder=None, true_labels_probs=None, encodings=None):
    prompts_discrete_indices = prompts_probs.argmax(1)

    if args.use_ngram:
        prompts_discrete_indices_ngram_list = []
        indices_list = prompts_discrete_indices.int().tolist()
        for idx in indices_list:
            prompts_discrete_indices_ngram_list.append(args.ngram_list[idx])
        prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)
    
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if args.trial and step >= 100:
                break
            bsz = len(batch['input_ids'])

            if args.use_ngram:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
            else:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)

            mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id)
            mask_pos = torch.tensor(mask_pos[-1])

            sequence_output = model(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"])
            # last_hidden_state = sequence_output[0].squeeze()
            last_hidden_state = sequence_output[0]
            logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

            label = batch["labels"].to(args.device)
            converted_target = label.clone()
            label_indices = [true_labels_probs[i].argmax(1) for i in range(args.n_class)]
            label_tokens = [encodings[i][label_indices[i]] for i in range(args.n_class)]
            interest_index = label_tokens
            logits = logits[:, interest_index]
            pred_label = logits.argmax(dim=-1)
            predictions_pred = torch.nn.functional.softmax(logits, dim=1)
            if args.task_name in {'mnli', 'snli'}:
                metric.add_batch(
                    predictions=accelerator.gather(predictions_pred.cpu().numpy()),
                    references=accelerator.gather(converted_target.cpu().numpy()),
                )
            else:
                metric.add_batch(
                    predictions=accelerator.gather(pred_label),
                    references=accelerator.gather(converted_target),
                )

    if args.file_name in DOMAIN_DATASET:
        eval_metric = metric.compute(average='macro')
    else:
        eval_metric = metric.compute()

    # print("** eval **")
    # print(f"epoch {epoch + 1}: {eval_metric}")
    if args.balance == True:
        eval_result = eval_metric['acc']
    else:
        eval_result = eval_metric['auc']
    write_results(args, folder, epoch, api_count, None, None, eval_metric, metric_state='eval')
    return eval_result

def blppg_test(args, model, test_dataloader, metric, accelerator, epoch, api_count, best_epoch, best_api_count, prompts_probs=None, prompt_length=None, tokenizer=None, test_dataloader_mm=None, folder=None, test_metric=None, true_labels_probs=None, encodings=None):
    if test_metric is not None:
        write_results(args, folder, epoch, api_count, best_epoch, best_api_count, test_metric, metric_state='test')
        return test_metric
    
    prompts_discrete_indices = prompts_probs.argmax(1)
    if args.use_ngram:
        prompts_discrete_indices_ngram_list = []
        indices_list = prompts_discrete_indices.int().tolist()
        for idx in indices_list:
            prompts_discrete_indices_ngram_list.append(args.ngram_list[idx])
        prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            if args.trial and step >= 100:
                break
            bsz = len(batch['input_ids'])
            
            if args.use_ngram:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
            else:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)

            mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id)
            mask_pos = torch.tensor(mask_pos[-1])

            sequence_output = model(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"])
            # last_hidden_state = sequence_output[0].squeeze()
            last_hidden_state = sequence_output[0]
            logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

            label = batch["labels"].to(args.device)
            converted_target = label.clone()
            label_indices = [true_labels_probs[i].argmax(1) for i in range(args.n_class)]
            label_tokens = [encodings[i][label_indices[i]] for i in range(args.n_class)]
            interest_index = label_tokens
            logits = logits[:, interest_index]
            pred_label = logits.argmax(dim=-1)
            predictions_pred = torch.nn.functional.softmax(logits, dim=1)
            if args.task_name in {'mnli', 'snli'}:
                metric.add_batch(
                    predictions=accelerator.gather(predictions_pred.cpu().numpy()),
                    references=accelerator.gather(converted_target.cpu().numpy()),
                )
            else:
                metric.add_batch(
                    predictions=accelerator.gather(pred_label),
                    references=accelerator.gather(converted_target),
                )
            
    if args.file_name in DOMAIN_DATASET:
        test_metric = metric.compute(average='macro')
    else:
        test_metric = metric.compute()

    # print("** test **")
    # print(f"current epoch {epoch + 1}, best epoch {best_epoch + 1}, api_count {api_count}: {test_metric}")
    if args.use_wandb:
        for key in test_metric.keys():
            eval_key = 'blppg_test_' + key
            wandb.log({eval_key: test_metric[key]})
    
    write_results(args, folder, epoch, api_count, best_epoch, best_api_count, test_metric, metric_state='test')
    return test_metric


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, tau=1.0):
    gumbel_noise = sample_gumbel(logits.shape)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1), gumbel_noise

def gumbel_softmax(logits, tau=1.0, hard=False):
    logits = torch.log(logits)
    y, gumbel_noise = gumbel_softmax_sample(logits, tau=tau)

    if hard:
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y

    return y, gumbel_noise

