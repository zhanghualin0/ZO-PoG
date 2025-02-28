import numpy as np
import torch
from common import DOMAIN_DATASET, write_results

TEMPLATE_CONFIG = {
    "mnli": " entailment?",
    "qqp": " equivalent?",
    "mrpc": " equivalent?",
    "cola": " correct?",
    "book": " It was ",
    "elec": " It was ",
    "wnli": " what is the relation?",
    "qnli": " entailment?",
    "rte": " entailment?",
    "CI": " What is the intent?",
    "SE": " What is the relation?",
    "RCT": " What is the role?",
    "HP": " Helpful?",
    "sst2": " It was ",
    "imdb": " It was ",
    "cr": " It was ",
    "snli": " entailment?",
    "agnews": " It was ",
}

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
                # batch['input_ids'] = torch.cat([tokenizer.bos_token_id + torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
                batch['input_ids'] = torch.cat([prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            else:
                # batch['input_ids'] = torch.cat([tokenizer.bos_token_id + torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
                batch['input_ids'] = torch.cat([prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            
            # batch["attention_mask"] = torch.cat([torch.ones(bsz, 1).to(args.device), torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)

            # mask_pos=np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
            # mask_pos = torch.tensor(mask_pos[-1])
            label_to_id = model.config.label2id 
            
            max_sentence_index = max(torch.sum(batch['input_ids'] != tokenizer.eos_token_id, dim=1))
            sequence_output = model(input_ids=batch['input_ids'][:, :max_sentence_index], attention_mask=batch["attention_mask"][:, :max_sentence_index])
            logits_out = sequence_output['logits']
            logits = logits_out[torch.arange(logits_out.size(0)), torch.sum(batch["attention_mask"], dim=1, dtype=int) - 1]
            # logits = sequence_output['logits'][:, -1]
            # print(logits)
            # last_hidden_state = sequence_output[0].squeeze()
            # logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

            label = batch["labels"].to(args.device)
            label_keys = list(label_to_id.keys())
            label_map = {}
            for target in label_keys:
                label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = label_to_id[target]
            converted_target = label.clone()
            for key, val in label_map.items():
                converted_target[label == key] = val
            sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
            interest_index = [item[0] for item in sort_label_map]  
            logits = logits[:, interest_index]
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
    eval_result = eval_metric['accuracy']
    write_results(args, folder, epoch, api_count, None, None, eval_metric, metric_state='eval')
    return eval_result

def test(args, model, test_dataloader, metric, accelerator, epoch, api_count, best_epoch, best_api_count, prompts_probs=None, prompt_length=None, tokenizer=None, test_dataloader_mm=None, folder=None, test_metric=None, tableName=None):
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
                # batch['input_ids'] = torch.cat([tokenizer.bos_token_id + torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
                batch['input_ids'] = torch.cat([prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            else:
                # batch['input_ids'] = torch.cat([tokenizer.bos_token_id + torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
                batch['input_ids'] = torch.cat([prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)

            # mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id)
            # mask_pos = torch.tensor(mask_pos[-1])
            label_to_id = model.config.label2id
            
            max_sentence_index = max(torch.sum(batch['input_ids'] != tokenizer.eos_token_id, dim=1))
            sequence_output = model(input_ids=batch['input_ids'][:, :max_sentence_index], attention_mask=batch["attention_mask"][:, :max_sentence_index])
            logits_out = sequence_output['logits']
            logits = logits_out[torch.arange(logits_out.size(0)), torch.sum(batch["attention_mask"], dim=1, dtype=int) - 1]
            # logits = sequence_output['logits'][:, -1]

            label = batch["labels"].to(args.device)
            label_keys = list(label_to_id.keys())
            label_map = {}
            for target in label_keys:
                label_map[tokenizer.encode(target, add_special_tokens=False)[0]]  = label_to_id[target]
            converted_target = label.clone()
            for key, val in label_map.items():
                converted_target[label == key] = val
            sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
            interest_index = [item[0] for item in sort_label_map]   
            logits = logits[:, interest_index]
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
        test_metric = metric.compute(average='macro')
    else:
        test_metric = metric.compute()

    # print("** test **")
    # print(f"current epoch {epoch + 1}, best epoch {best_epoch + 1}, api_count {api_count}: {test_metric}")
    if args.use_wandb:
        for key in test_metric.keys():
            eval_key = 'Black_test_' + key
            wandb.log({eval_key: test_metric[key]})
    
    write_results(args, folder, epoch, api_count, best_epoch, best_api_count, test_metric, metric_state='test', tableName=tableName)
    return test_metric

def get_batch_indices(start, end, indices):
    batch_indices = np.where((indices >= start) & (indices < end))[0]
    if len(batch_indices) > 0:
        batch_indices = indices[batch_indices] - start
    return batch_indices

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
                batch['input_ids'] = torch.cat([prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            else:
                batch['input_ids'] = torch.cat([prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)

            label_to_id = model.config.label2id 
            
            max_sentence_index = max(torch.sum(batch['input_ids'] != tokenizer.eos_token_id, dim=1))
            sequence_output = model(input_ids=batch['input_ids'][:, :max_sentence_index], attention_mask=batch["attention_mask"][:, :max_sentence_index])
            logits_out = sequence_output['logits']
            logits = logits_out[torch.arange(logits_out.size(0)), torch.sum(batch["attention_mask"], dim=1, dtype=int) - 1]

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
                batch['input_ids'] = torch.cat([prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            else:
                batch['input_ids'] = torch.cat([prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)

            label_to_id = model.config.label2id 
            
            max_sentence_index = max(torch.sum(batch['input_ids'] != tokenizer.eos_token_id, dim=1))
            sequence_output = model(input_ids=batch['input_ids'][:, :max_sentence_index], attention_mask=batch["attention_mask"][:, :max_sentence_index])
            logits_out = sequence_output['logits']
            logits = logits_out[torch.arange(logits_out.size(0)), torch.sum(batch["attention_mask"], dim=1, dtype=int) - 1]

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
            eval_key = 'Black_test_' + key
            wandb.log({eval_key: test_metric[key]})
    
    write_results(args, folder, epoch, api_count, best_epoch, best_api_count, test_metric, metric_state='test')
    return test_metric
