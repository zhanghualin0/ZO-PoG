# Copyright 2023 shizhediao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from collections import defaultdict
import re
import math
import numpy
import os

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
)
from modelscope import snapshot_download
from transformers import LlamaForCausalLM
os.environ['TRANSFORMERS_OFFLINE']="1"
data_name='agnews'
model='llama3'
assert model in ['roberta-large', 'gpt2-xl', 'llama3']
model_cache_dir = '/root/autodl-tmp/newp3l/' + model
if model=='roberta-large':
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=True, cache_dir=model_cache_dir)
elif model=='gpt2-xl':
    tokenizer = AutoTokenizer.from_pretrained('gpt2-xl', use_fast=True, cache_dir=model_cache_dir)
elif model=='llama3':
    model_cache_dir = '/root/autodl-tmp/newp3l/' + 'LLM-Research/Meta-Llama-3-8B'
    model_id = snapshot_download('LLM-Research/Meta-Llama-3-8B', cache_dir=model_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=model_cache_dir)

class FindNgrams:
    def __init__(self, min_count=0, min_pmi=0, language='en'):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.words = defaultdict(int)
        self.ngrams, self.pairs = defaultdict(int), defaultdict(int)
        self.total = 0.
        self.language = language

    def text_filter(self, sentence):
        cleaned_text = []
        index = 0
        for i, w in enumerate(sentence):
            if re.match(u'[^\u0600-\u06FF\u0750-\u077F\u4e00-\u9fa50-9a-zA-Z]+', w):
                if i > index:
                    cleaned_text.append([w.lower() for w in sentence[index:i]])
                index = 1 + i
        if index < len(sentence):
            cleaned_text.append([w.lower() for w in sentence[index:]])
        return cleaned_text

    def count_ngram(self, texts, n):
        self.ngrams = defaultdict(int)
        for sentence in texts:
            sub_sentence = sentence.split()
            for i in range(n):
                n_len = i + 1
                for j in range(len(sub_sentence) - i):
                    ngram = tuple([w for w in sub_sentence[j: j+n_len]])
                    self.ngrams[ngram] += 1
        self.ngrams = {i:j for i, j in self.ngrams.items() if j > self.min_count}

    def find_ngrams_pmi(self, texts, n, freq_threshold):
        for sentence in texts:
            # print(sentence)
            sub_sentence = sentence.split()
            self.words[sub_sentence[0]] += 1
            for i in range(len(sub_sentence)-1):
                self.words[sub_sentence[i + 1]] += 1
                self.pairs[(sub_sentence[i], sub_sentence[i+1])] += 1
                self.total += 1
        self.words = {i: j for i, j in self.words.items() if j > self.min_count}
        self.pairs = {i: j for i, j in self.pairs.items() if j > self.min_count}

        min_mi = math.inf
        max_mi = -math.inf

        self.strong_segments = set()
        for i, j in self.pairs.items():
            if i[0] in self.words and i[1] in self.words:
                mi = math.log(self.total * j / (self.words[i[0]] * self.words[i[1]]))
                if mi > max_mi:
                    max_mi = mi
                if mi < min_mi:
                    min_mi = mi
                if mi >= self.min_pmi:
                    self.strong_segments.add(i)


        self.ngrams = defaultdict(int)
        for sentence in texts:
            sub_sentence = sentence.split()
            s = [sub_sentence[0]]
            for i in range(len(sub_sentence)-1):
                if (sub_sentence[i], sub_sentence[i+1]) in self.strong_segments:
                    s.append(sub_sentence[i+1])
                else:
                    self.ngrams[tuple(s)] += 1
                    s = [sub_sentence[i+1]]
        self.ngrams = {i:j for i, j in self.ngrams.items() if j > self.min_count and len(i) <= n}

        self.renew_ngram_by_freq(texts, freq_threshold, n)


    def renew_ngram_by_freq(self, all_sentences, min_feq, ngram_len=10):
        new_ngram2count = {}
        new_all_sentences = []

        for sentence in all_sentences:
            sentence = sentence.split()
            sen = sentence
            for i in range(len(sen)):
                for n in range(1, ngram_len + 1):
                    if i + n > len(sentence):
                        break
                    n_gram = tuple(sentence[i: i + n])
                    if n_gram not in self.ngrams:
                        continue
                    if n_gram not in new_ngram2count:
                        new_ngram2count[n_gram] = 1
                    else:
                        new_ngram2count[n_gram] += 1
        self.ngrams = {gram: c for gram, c in new_ngram2count.items() if c > min_feq}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=data_name, type=str, help="the name of dataset") #改名
parser.add_argument('--output_dir', type=str, help="the output path", default=f'./pmi/{model}/{model}_{data_name}.txt')
parser.add_argument('--ngram', type=int, default=5, help="n")
parser.add_argument('--min_count', type=int, default=5, help="min_count")
parser.add_argument('--min_pmi', type=int, default=1, help="min_pmi")
parser.add_argument('--ngram_freq_threshold', type=int, default=5, help="ngram_freq_threshold")
parser.add_argument('--delete_special_symbol', action='store_false', help="Whether to remove special symbols")
args = parser.parse_args()

ngram_list = []
dataset = args.dataset
ngram = args.ngram
min_count = args.min_count
min_pmi = args.min_pmi
ngram_freq_threshold = args.ngram_freq_threshold

print('dataset: ', dataset)
# f_read = open(args.dataset, 'r')
# cache_dir = '/home/public/BoPromptData/' + args.dataset
# if not os.path.exists(cache_dir):
#     os.makedirs(cache_dir)
# raw_datasets = load_dataset('glue', args.dataset, cache_dir=cache_dir)
args.train_file = './dataset/' + args.dataset + '/train.csv' if args.dataset else None
args.validation_file = './dataset/' + args.dataset + '/dev.csv' if args.dataset else None
args.test_file = './dataset/' + args.dataset + '/test.csv' if args.dataset else None

if args.dataset is not None:
    if args.dataset in ["book", "elec"]:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        cache_dir = '/root/autodl-tmp/BoPromptData/' + args.dataset
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if args.dataset == 'snli':
            raw_datasets = load_dataset(args.dataset, cache_dir=cache_dir)
            # raw_datasets = load_dataset(args.dataset)
            raw_datasets = raw_datasets.filter(lambda example: example['label'] in [0, 1, 2])
        elif args.dataset == 'agnews':
            raw_datasets = load_dataset('ag_news', 'default', cache_dir=cache_dir)
        else:
            raw_datasets = load_dataset('glue', args.dataset, cache_dir=cache_dir)
            #raw_datasets = load_dataset("glue", args.dataset)
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

f_read = raw_datasets["train"]
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "book": ("sentence", None),
    "elec": ("sentence", None),
    "snli": ("premise", "hypothesis"),
    "agnews": ("text", None),
}
sentence1_key, sentence2_key = task_to_keys[args.dataset]
sentence_list = []
if sentence2_key:
    for s1, s2 in zip(f_read[sentence1_key], f_read[sentence2_key]):
        if s1 != None and s2 != None:
            sentence_list.append(s1+' '+s2)
else:
    for s1 in f_read[sentence1_key]:
        if s1 != None:
            sentence_list.append(s1)
f_write = open(args.output_dir, 'w')

# sentence_list = []
# for line in f_read:
#     sentence_list.append(line)

ngram_finder = FindNgrams(min_count=min_count, min_pmi=min_pmi)
ngram_finder.find_ngrams_pmi(sentence_list, ngram, ngram_freq_threshold)

ngram_type_count = [0 for _ in range(ngram)]
ngram_finder.ngrams = dict(sorted(ngram_finder.ngrams.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))  #sort


count = 0
for w, c in ngram_finder.ngrams.items():
    count += 1
    s = ""
    for word_index in range(len(w)):
        s += w[word_index]+" "
    s = s.strip()
    i = len(s)
    if args.delete_special_symbol:
        while i>0:
            if s[i-1].isalnum():
                break
            i -= 1
    s = s[0:i]
    if s not in ngram_list and len(s)>0:
        if s not in ngram_list:
            ngram_list.append(s)

ngram_count = 0
for ngram_phrase in ngram_list:
    ngram_index = tokenizer.encode(' ' + ngram_phrase, add_special_tokens=False)
    ngram_index_list = list(map(str, ngram_index))
    ngram_index_str = ",".join(ngram_index_list)
    ngram_count += 1
    # f_write.write(ngram_phrase + '\n') # for GPT
    f_write.write(ngram_index_str +'\n') # for RoBERTa
    ngram_type_count[len(list(ngram_phrase.split())) - 1] += 1

print(str(ngram_type_count))
#f_read.close()
f_write.close()