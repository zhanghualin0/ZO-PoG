import os
import copy
import time
import random

import torch
# import fitlog
import argparse
import numpy as np
import cma

from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    BertConfig,
    BertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    BartConfig,
    BartTokenizer,
    T5Config,
    T5Tokenizer,
    GPT2Config,
    AutoConfig,
    GPT2Tokenizer,
    BartConfig as CPTConfig,
)
from models.modeling_roberta import RobertaForMaskedLM
from models.modeling_bart import BartForConditionalGeneration
from models.modeling_t5 import T5ForConditionalGeneration
from models.modeling_gpt2 import GPT2LMHeadModel
from models.modeling_bert import BertForMaskedLM
from models.modeling_electra import ElectraForMaskedLM
from models.modeling_cpt import CPTForMaskedLM
from models.modeling_llama3 import LlamaForCausalLM
from utils import hinge_loss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from fastNLP.core.metrics import Metric as MetricBase
from common import LABEL2ID_CONFIG, Imbalanced_Metric
from modelscope import snapshot_download

class Metric(MetricBase):
    def __init__(self, tokenizer, label_map, pred=None, target=None, seq_len=None):
        super().__init__()
        # Initialize parameter mapping manually since _init_param_map doesn't exist in fastNLP 1.0.1
        self.pred = pred
        self.target = target
        self.seq_len = seq_len
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        self.label_map = label_map
        # = {
        #     tokenizer.encode('bad', add_special_tokens=False)[0]: 0,  # negative
        #     tokenizer.encode('great', add_special_tokens=False)[0]: 1,  # positive
        # }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target==key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)


    def get_metric(self, reset=True):
        labels = self._target
        preds = self._pred
        auc = roc_auc_score(y_true=labels, y_score=preds, average="macro")
        acc = accuracy_score(y_true=labels, y_pred=preds)
        f1 = float(f1_score(y_true=labels, y_pred=preds))
        precision = float(precision_score(y_true=labels, y_pred=preds))
        recall = float(recall_score(y_true=labels, y_pred=preds))
        m_corrcoef = matthews_corrcoef(y_true=labels, y_pred=preds)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'auc': auc, 
                'acc': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'm_corrcoef': m_corrcoef,
                'hinge': hinge_loss,
                'ce': ce_loss}

class LMForwardAPI:
    def __init__(self, args, model_name='roberta-large', task_name='sst2',
                 loss_type='hinge', init_prompt_path=None, test_dataloader=None, eval_dataloader=None, tokenizer=None):
        self.n_prompt_tokens=args.n_prompt_tokens
        if model_name in ['roberta-base', 'roberta-large']:
            model_cache_dir = './download/model/' + model_name
            if not os.path.exists(model_cache_dir):
                os.makedirs(model_cache_dir)
            self.config = RobertaConfig.from_pretrained(model_name, cache_dir=model_cache_dir)
            # self.tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
            self.model = RobertaForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=self.n_prompt_tokens,
                inference_framework='pt',
                onnx_model_path=None,
                cache_dir=model_cache_dir
            )
            self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.config.vocab_size))
        elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
            self.config = BertConfig.from_pretrained(model_name)
            # self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=self.n_prompt_tokens,
            )
        elif model_name in ['google/electra-base-generator', 'google/electra-large-generator']:
            self.config = ElectraConfig.from_pretrained(model_name)
            # self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
            self.model = ElectraForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=self.n_prompt_tokens,
            )
        elif model_name in ['facebook/bart-base', 'facebook/bart-large']:
            self.config = BartConfig.from_pretrained(model_name)
            # self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=self.n_prompt_tokens,
            )
        elif model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
            self.config = T5Config.from_pretrained(model_name)
            # self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=self.n_prompt_tokens,
            )
        elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            model_cache_dir = './download/model/' + model_name
            if not os.path.exists(model_cache_dir):
                os.makedirs(model_cache_dir)
            self.config = GPT2Config.from_pretrained(model_name, cache_dir=model_cache_dir)
            # self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=self.n_prompt_tokens,
                cache_dir=model_cache_dir
            )
        elif model_name in ['llama3',]:
            model_cache_dir = './download/model/' + 'LLM-Research/Meta-Llama-3-8B'
            pretrained_model_name_or_path = snapshot_download('LLM-Research/Meta-Llama-3-8B', cache_dir=model_cache_dir)
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path, cache_dir=model_cache_dir)
            self.model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, config=self.config, n_prompt_tokens=self.n_prompt_tokens, cache_dir=model_cache_dir)
        elif model_name in ['fnlp/cpt-large']:
            self.config = CPTConfig.from_pretrained(model_name)
            # self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = CPTForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=self.n_prompt_tokens,
            )
        else:
            raise NotImplementedError
        self.tokenizer = tokenizer
        if args.cat_or_add == 'cat':
            self.model.set_concat_prompt(True)
            if init_prompt_path is not None:
                print('Initialize prompt embedding from {}'.format(init_prompt_path))
                self.init_prompt = torch.load(init_prompt_path).weight.cpu().reshape(-1)
            else:
                print('Initial prompt embedding not found. Initialize to zero embedding.')
                self.init_prompt = torch.zeros(self.n_prompt_tokens * self.config.hidden_size)
            print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))
        else:
            # self.model.set_concat_prompt(False)
            self.init_prompt = None
        self.model.to(args.device)
        self.model.eval()
        self.linear = torch.nn.Linear(args.intrinsic_dim, self.n_prompt_tokens * self.config.hidden_size, bias=False).to(args.device)
        if args.random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['roberta-base', 'roberta-large']:
                embedding = self.model.roberta.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
                embedding = self.model.bert.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['google/electra-base-generator', 'google/electra-large-generator']:
                embedding = self.model.electra.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['facebook/bart-base', 'facebook/bart-large', 'fnlp/cpt-large']:
                embedding = self.model.model.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                embedding = self.model.transformer.get_input_embeddings().weight.clone().cpu()
            else:  # T5, llama3
                embedding = self.model.get_input_embeddings().weight.clone().cpu()
            # embedding = embedding[1000: 2000]
            mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = args.alpha * std_hat / (np.sqrt(args.intrinsic_dim) * args.sigma)
            # temp = intrinsic_dim - std_hat * std_hat
            # mu = mu_hat / temp
            # std = std_hat / np.sqrt(temp)
            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():
                torch.nn.init.normal_(p, mu, std)
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_prompt = None
        self.num_call = 0
        # self.save_path = save_path
        self.print_every = args.print_every
        self.eval_every = args.eval_every
        self.loss_type = args.loss_type
        # if save_path is not None:
        #     os.makedirs(save_path, exist_ok=True)
        label_to_id = LABEL2ID_CONFIG[task_name]
        label_keys = list(label_to_id.keys())
        self.num_labels = len(label_to_id)
        label_map = {}
        for target in label_keys:
            label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = label_to_id[target]
        self.metric = Metric(tokenizer, label_map, pred='logits', target='labels',)
        self.metric_name = f'{task_name}Metric'
        if args.balance:
            self.metric_key = 'acc'
        else:
            self.metric_key = 'auc'
        print(f"*******************self.metric_key={self.metric_key}******************")
        self.margin = self.metric.margin
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.per_device_eval_batch_size = args.per_device_eval_batch_size

        self.best_api_count = 0
        self.best_probs = None

    def calc_metric(self, logits, target):
        label_map = self.metric.label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
        interest_index = [item[0] for item in sort_label_map]
        logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1)

        if self.metric_key == 'acc':
            perf = (pred == converted_target).sum() / len(target)
        elif self.metric_key == 'auc':
            if self.num_labels == 2:
                perf = roc_auc_score(y_true=converted_target.detach().cpu().numpy().tolist(), y_score=pred.detach().cpu().numpy().tolist(), average="macro")
            else:
                predictions_pred = torch.nn.functional.softmax(logits, dim=1)
                perf = roc_auc_score(y_true=converted_target, y_score=predictions_pred, multi_class='ovr')
        elif self.metric_key == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(),
                            pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [auc, f1], got {self.metric_key} instead.')

        if self.loss_type == 'hinge':
            loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
        elif self.loss_type == 'ce':
            loss = self.ce_loss(logits, converted_target).item()
        elif self.loss_type == 'perf':
            loss = -1 * perf
        else:
            raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.')

        return loss, perf

    def eval(self, model_name, imMetric, prompt_embedding=None, test_dataloader=None):
        # self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        bsz = len(self.train_dataloader)
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
                if self.init_prompt is not None:
                    z = z + self.init_prompt  # Az + p_0
                pe_list.append(z.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1))
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
            assert len(prompt_embedding) == len(train_data['input_ids'])
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        self.model.set_prompt_embedding(prompt_embedding)
        
        def get_loss_pref(model_name, imMetric, batch_data, flag='train'):
            with torch.no_grad():
                if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        decoder_input_ids=batch_data['decoder_input_ids'],
                        decoder_attention_mask=batch_data['decoder_attention_mask'],
                    )['logits']
                elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama3']:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                    )['logits']
                    assert sum(batch_data['input_ids'][0]!=self.tokenizer.eos_token_id) == sum(batch_data["attention_mask"][0])
                    max_sentence_index = max(torch.sum(batch_data['input_ids'] != self.tokenizer.eos_token_id, dim=1))
                    logits = logits[torch.arange(logits.size(0)), torch.sum(batch_data['attention_mask'][:, :max_sentence_index], dim=1, dtype=int) - 1]
                else:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        mask_pos=batch_data['mask_pos'],
                    )['logits']
                    
                target = batch_data['labels']
                label_map = self.metric.label_map
                converted_target = target.clone()
                for key, val in label_map.items():
                    converted_target[target == key] = val
                sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
                interest_index = [item[0] for item in sort_label_map]
                logits = logits[:, interest_index]
                pred = logits.argmax(dim=-1)
                
                if self.num_labels > 2:
                    predictions_pred = torch.nn.functional.softmax(logits, dim=1)
                    imMetric.add_batch(predictions=predictions_pred, references=converted_target)
                else:
                    imMetric.add_batch(predictions=pred, references=converted_target)
            
                if flag == 'train':
                    if self.loss_type == 'hinge':
                        loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
                    elif self.loss_type == 'ce':
                        loss = self.ce_loss(logits, converted_target).item()
                    elif self.loss_type == 'perf':
                        loss = -1 * perf
                    return loss

        if test_dataloader:
            for step, test_data in enumerate(test_dataloader):
                get_loss_pref(model_name, imMetric, test_data, flag='test')
            test_metric = imMetric.compute()
            print(f'On Test [# API Calls {self.num_call}] Metric {test_metric}')
            return test_metric
        else:
            loss_list = []
            for step, train_data in enumerate(self.train_dataloader):
                self.num_call += 1
                loss = get_loss_pref(model_name, imMetric, train_data)
                loss_list.append(loss)
            loss_avg = np.mean(loss_list)
            
            train_metric = imMetric.compute()
            print(f'On Training [# API Calls {self.num_call}] Loss avg {loss_avg} Metric {train_metric}')

            print('********* Evaluated on dev set *********')
            for step, eval_data in enumerate(self.eval_dataloader):
                # self.num_call += 1
                get_loss_pref(model_name, imMetric, eval_data, flag='eval')
            
            eval_metric = imMetric.compute()
            print(f'On Eval {eval_metric}')
            if eval_metric[self.metric_key] > self.best_dev_perf:
                self.best_dev_perf = eval_metric[self.metric_key]
                self.best_prompt = copy.deepcopy(tmp_prompt)
                
        return loss_avg, self.num_call

    def my_eval(self, model_name, imMetric, prompt_embedding=None, data=None):
        # self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        bsz = len(data['input_ids'])
        #tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if torch.is_tensor(prompt_embedding):
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            #prompt_embedding = prompt_embedding.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1)
            prompt_embedding = prompt_embedding.reshape(bsz, self.n_prompt_tokens, -1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        self.model.set_prompt_embedding(prompt_embedding)
        
        def get_loss_pref(model_name, imMetric, batch_data, flag='train'):
            with torch.no_grad():
                if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        decoder_input_ids=batch_data['decoder_input_ids'],
                        decoder_attention_mask=batch_data['decoder_attention_mask'],
                    )['logits']
                elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama3']:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                    )['logits']
                    assert sum(batch_data['input_ids'][0]!=self.tokenizer.eos_token_id) == sum(batch_data["attention_mask"][0])
                    max_sentence_index = max(torch.sum(batch_data['input_ids'] != self.tokenizer.eos_token_id, dim=1))
                    logits = logits[torch.arange(logits.size(0)), torch.sum(batch_data['attention_mask'][:, :max_sentence_index], dim=1, dtype=int) - 1]
                else:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        mask_pos=batch_data['mask_pos'],
                    )['logits']
                    
                target = batch_data['labels']
                label_map = self.metric.label_map
                converted_target = target.clone()
                for key, val in label_map.items():
                    converted_target[target == key] = val
                sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
                interest_index = [item[0] for item in sort_label_map]
                logits = logits[:, interest_index]
                #pred = logits.argmax(dim=-1)
                #imMetric.add_batch(predictions=pred, references=converted_target)
            
                if flag == 'train':
                    with torch.enable_grad():
                        if self.loss_type == 'hinge':
                            loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
                        elif self.loss_type == 'ce':
                            loss = self.ce_loss(logits, converted_target)
                        elif self.loss_type == 'auc':
                            probs = torch.nn.functional.softmax(logits, dim=1)
                            probs = probs[:, 1]
                            loss = self.auc_loss(probs, converted_target, auto=False)
                            #torch.mean(loss).backward()
                        return loss

        self.num_call += 1
        loss = get_loss_pref(model_name, imMetric, data)
        
        #train_metric = imMetric.compute()
        #print(f'On Training [# API Calls {self.num_call}] Loss avg {torch.mean(loss)} Metric {train_metric}')
                
        return loss, self.num_call
    
    def my_eval2(self, model_name, imMetric, prompt_embedding=None, data=None):
        # self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        bsz = len(data['input_ids'])
        #tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if torch.is_tensor(prompt_embedding):
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            #prompt_embedding = prompt_embedding.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1)
            prompt_embedding = prompt_embedding.reshape(bsz, self.n_prompt_tokens, -1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        self.model.set_prompt_embedding(prompt_embedding)
        
        def get_loss_pref(model_name, imMetric, batch_data, flag='train'):
            with torch.no_grad():
                if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        decoder_input_ids=batch_data['decoder_input_ids'],
                        decoder_attention_mask=batch_data['decoder_attention_mask'],
                    )['logits']
                elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama3']:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                    )['logits']
                else:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        mask_pos=batch_data['mask_pos'],
                    )['logits']
                    
                target = batch_data['labels']
                label_map = self.metric.label_map
                converted_target = target.clone()
                for key, val in label_map.items():
                    converted_target[target == key] = val
                sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
                interest_index = [item[0] for item in sort_label_map]
                logits = logits[:, interest_index]
                #pred = logits.argmax(dim=-1)
                #imMetric.add_batch(predictions=pred, references=converted_target)
            
                if flag == 'train':
                    with torch.enable_grad():
                        if self.loss_type == 'hinge':
                            loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
                        elif self.loss_type == 'ce':
                            loss = self.ce_loss(logits, converted_target)
                        elif self.loss_type == 'auc':
                            probs = torch.nn.functional.softmax(logits, dim=1)
                            probs = probs[:, 1]
                            loss = self.auc_loss(probs, converted_target, auto=False)
                            torch.mean(loss).backward()
                        return loss

        self.num_call += 1
        loss = get_loss_pref(model_name, imMetric, data)
        
        #train_metric = imMetric.compute()
        #print(f'On Training [# API Calls {self.num_call}] Loss avg {torch.mean(loss)} Metric {train_metric}')
                
        return loss, self.num_call
    

    def my_dev_eval(self, model_name, imMetric, prompt_embedding=None, prompts_probs=None, args=None):
        #if self.num_call % self.eval_every == 0:
        print('********* Evaluated on dev set *********')

        bsz = self.per_device_eval_batch_size
        tmp_prompt = copy.deepcopy(prompt_embedding)
        if torch.is_tensor(prompt_embedding):
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        self.model.set_prompt_embedding(prompt_embedding)
        
        def get_loss_pref(model_name, imMetric, batch_data, flag='train'):
            with torch.no_grad():
                if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        decoder_input_ids=batch_data['decoder_input_ids'],
                        decoder_attention_mask=batch_data['decoder_attention_mask'],
                    )['logits']
                elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama3']:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                    )['logits']
                    assert sum(batch_data['input_ids'][0]!=self.tokenizer.eos_token_id) == sum(batch_data["attention_mask"][0])
                    max_sentence_index = max(torch.sum(batch_data['input_ids'] != self.tokenizer.eos_token_id, dim=1))
                    logits = logits[torch.arange(logits.size(0)), torch.sum(batch_data['attention_mask'][:, :max_sentence_index], dim=1, dtype=int) - 1]
                else:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        mask_pos=batch_data['mask_pos'],
                    )['logits']
                    
                target = batch_data['labels']
                label_map = self.metric.label_map
                converted_target = target.clone()
                for key, val in label_map.items():
                    converted_target[target == key] = val
                sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
                interest_index = [item[0] for item in sort_label_map]
                logits = logits[:, interest_index]
                pred = logits.argmax(dim=-1)

                if self.num_labels > 2:
                    predictions_pred = torch.nn.functional.softmax(logits, dim=1)
                    imMetric.add_batch(predictions=predictions_pred, references=converted_target)
                else:
                    imMetric.add_batch(predictions=pred, references=converted_target)
            
                if flag == 'train':
                    if self.loss_type == 'hinge':
                        loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
                    elif self.loss_type == 'ce':
                        loss = self.ce_loss(logits, converted_target)
                    elif self.loss_type == 'auc':
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        probs = probs[:, 1]
                        loss = self.auc_loss(probs, converted_target, auto=False)
                        #loss.backward()
                    return loss

        for step, eval_data in enumerate(self.eval_dataloader):
            if model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama3']:
                new_eval_data = update_p0_gpt(args, eval_data, prompts_probs)
            else:
                new_eval_data = update_p0(args, eval_data, prompts_probs)
            get_loss_pref(model_name, imMetric, new_eval_data, flag='eval')
            #get_loss_pref(model_name, imMetric, eval_data, flag='eval')
        
        eval_metric = imMetric.compute()
        print(f'Query Numbers: {self.num_call}. On Eval {eval_metric}')
        if eval_metric[self.metric_key] > self.best_dev_perf:
            self.best_dev_perf = eval_metric[self.metric_key]
            self.best_api_count = copy.deepcopy(self.num_call)
            self.best_prompt = copy.deepcopy(tmp_prompt)
            self.best_probs = copy.deepcopy(prompts_probs)
        print(f'best query numbers: {self.best_api_count}, best dev acc: {self.best_dev_perf}')


    def my_test_eval(self, model_name, imMetric, args=None):
        print('********* Evaluated on test set *********')

        bsz = self.per_device_eval_batch_size
        prompt_embedding = self.best_prompt
        prompts_probs = self.best_probs
        if torch.is_tensor(prompt_embedding):
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        self.model.set_prompt_embedding(prompt_embedding)
        
        def get_loss_pref(model_name, imMetric, batch_data, flag='train'):
            with torch.no_grad():
                if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        decoder_input_ids=batch_data['decoder_input_ids'],
                        decoder_attention_mask=batch_data['decoder_attention_mask'],
                    )['logits']
                elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama3']:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                    )['logits']
                    assert sum(batch_data['input_ids'][0]!=self.tokenizer.eos_token_id) == sum(batch_data["attention_mask"][0])
                    max_sentence_index = max(torch.sum(batch_data['input_ids'] != self.tokenizer.eos_token_id, dim=1))
                    logits = logits[torch.arange(logits.size(0)), torch.sum(batch_data['attention_mask'][:, :max_sentence_index], dim=1, dtype=int) - 1]
                else:
                    logits = self.model(
                        input_ids=batch_data['input_ids'],
                        attention_mask=batch_data['attention_mask'],
                        mask_pos=batch_data['mask_pos'],
                    )['logits']
                    
                target = batch_data['labels']
                label_map = self.metric.label_map
                converted_target = target.clone()
                for key, val in label_map.items():
                    converted_target[target == key] = val
                sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
                interest_index = [item[0] for item in sort_label_map]
                logits = logits[:, interest_index]
                pred = logits.argmax(dim=-1)

                if self.num_labels > 2:
                    predictions_pred = torch.nn.functional.softmax(logits, dim=1)
                    imMetric.add_batch(predictions=predictions_pred, references=converted_target)
                else:
                    imMetric.add_batch(predictions=pred, references=converted_target)
            
                if flag == 'train':
                    if self.loss_type == 'hinge':
                        loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
                    elif self.loss_type == 'ce':
                        loss = self.ce_loss(logits, converted_target)
                    elif self.loss_type == 'auc':
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        probs = probs[:, 1]
                        loss = self.auc_loss(probs, converted_target, auto=False)
                    return loss

                return logits, converted_target

        test_logits = []
        test_labels = []
        for step, test_data in enumerate(self.test_dataloader):
            if model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama3']:
                new_test_data = update_p0_gpt(args, test_data, prompts_probs)
            else:
                new_test_data = update_p0(args, test_data, prompts_probs)
            logits_, converted_target_=get_loss_pref(model_name, imMetric, new_test_data, flag='test')
            #logits_, converted_target_=get_loss_pref(model_name, imMetric, test_data, flag='test')
            test_logits.append(logits_)
            test_labels.append(converted_target_)
        test_logits = torch.cat(test_logits)
        test_labels = torch.cat(test_labels)
        test_metric = imMetric.compute()
        print(f'On Test [# Best API Calls {self.best_api_count}] Metric {test_metric}')
        return test_metric, test_logits, test_labels


def update_p0(args, batch, prompts_probs, indices_list=None):
    bsz = len(batch['input_ids'])
    prompts_discrete_indices = prompts_probs.argmax(1)

    new_batch = copy.deepcopy(batch)

    if args.use_ngram:
        prompts_discrete_indices_ngram_list = []
        if indices_list is None:
            indices_list = prompts_discrete_indices.int().tolist()
        for idx in indices_list:
            prompts_discrete_indices_ngram_list.append(args.ngram_list[idx])
        prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)
        assert prompts_discrete_ngram_indices.shape[-1] == args.n_prompt_tokens
        new_batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, (1+args.n_prompt_tokens):]], dim=1)

    return new_batch

def update_p0_gpt(args, batch, prompts_probs, indices_list=None):
    bsz = len(batch['input_ids'])
    prompts_discrete_indices = prompts_probs.argmax(1)

    new_batch = copy.deepcopy(batch)

    if args.use_ngram:
        prompts_discrete_indices_ngram_list = []
        if indices_list is None:
            indices_list = prompts_discrete_indices.int().tolist()
        for idx in indices_list:
            prompts_discrete_indices_ngram_list.append(args.ngram_list[idx])
        prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)
        assert prompts_discrete_ngram_indices.shape[-1] == args.n_prompt_tokens
        new_batch['input_ids'] = torch.cat([prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, (args.n_prompt_tokens):]], dim=1)

    return new_batch