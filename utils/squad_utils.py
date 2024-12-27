import os
import re
import string
import requests
import json
import random
import torch
import numpy as np
import collections


def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(False)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def read_data(path):
    
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    #dict to store contexts, questions, and answers.
    datas = {'question': [], 'id': [], 'answers': [], 'is_impossible': [], 'context': []}

    #iterate through all data in squad.
    for data in squad_dict['data']:
        for paragraph in data['paragraphs']:
            context = paragraph['context'].strip()
            for qa in paragraph['qas']:
                question = qa['question'].strip()
                
                datas['question'].append(question)
                datas['id'].append(qa['id'])
                datas['answers'].append(qa['answers'])
                datas['is_impossible'].append(qa['is_impossible'])
                datas['context'].append(context)
                
    #return formatted lists of data.
    return datas

def encode(data, tokenizer, max_length, doc_stride):
    encodings = tokenizer(data['question'],
                          data['context'],
                          max_length=max_length,
                          truncation='only_second',
                          stride=doc_stride,
                          return_overflowing_tokens=True,
                          return_offsets_mapping=True,
                          padding='max_length')
    return encodings

def add_position_tokens(data, encodings, sample_mapping, offset_mapping):
    start_positions = []
    end_positions = []

    for i, mapping_idx in enumerate(sample_mapping):
        start_pos = []
        end_pos = []
        answer = data['answers'][mapping_idx]
        offset = offset_mapping[i]
        if len(answer): #has an answer.
            answer = answer[0] #training data has at most 1 answer for each question.
            start_char = answer['answer_start']
            end_char = start_char + len(answer['text'])
            sequence_ids = encodings.sequence_ids(i)

            #find the start and end of the answer in context.
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            #if the answer is not fully inside the context, label it (0,0).
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                #otherwise it's the start and end token positions.
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        else: #no answers, label with (0,0).
            start_positions.append(0)
            end_positions.append(0)
            
    encodings['start_positions'] = start_positions
    encodings['end_positions'] = end_positions

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
    

def process_predictions(raw_predictions, test_data, test_encodings, tokenizer, test_sample_mapping, test_offset_mapping, n_best_size=20, max_answer_length=30):
    predictions = []

    i = 0
    while i < len(raw_predictions):
        valid_answers = []
        min_null_score = None #if is_impossible, the correct answer is set to (0,0), this will compare the score at (0,0) with the best score produced.
        start_logits = []
        end_logits = []
        offset_mappings = []
        sequence_ids = []

        idx = 0 #store the number of features map to this test_data['answers'].
        #store all the logits belong to this mapping index in test_data['answers'].
        while test_sample_mapping[i+idx]==test_sample_mapping[i]:
            start_logits.append(raw_predictions[i+idx][0].cpu().numpy()) #get all start logits.
            end_logits.append(raw_predictions[i+idx][1].cpu().numpy()) #get all end logits.
            offset_mappings.append(test_offset_mapping[i+idx]) #get all offsets.
            sequence_ids.append(test_encodings.sequence_ids(i+idx))
            idx += 1
            if i + idx >= len(raw_predictions): #the very last iteration.
                break
        
        #go through the features map to this test_data['answers'].
        for j in range(idx):
            #update minimum null prediction.
            cls_index = test_encodings['input_ids'][test_sample_mapping[i+j]].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[j][cls_index] + end_logits[j][cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            #a list of possible start/end indexes.
            start_indexes = np.argsort(start_logits[j])[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits[j])[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    #out of length or not in context indexes.
                    if (start_index >= len(offset_mappings[j]) or 
                        end_index >= len(offset_mappings[j]) or 
                        sequence_ids[j][start_index] != 1 or 
                        sequence_ids[j][end_index] != 1):
                        continue
                    #negative length or length greater than the set max length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mappings[j][start_index][0]
                    end_char = offset_mappings[j][end_index][1]
                    valid_answers.append({'score': start_logits[j][start_index] + end_logits[j][end_index],
                                          'text': test_data['context'][test_sample_mapping[i+j]][start_char:end_char]})
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x['score'], reverse=True)[0]
        else:
            best_answer = {'text': "", 'score': 0.0} #dummy for rare edge case.

        predictions.append(best_answer['text'] if best_answer['score'] > min_null_score else "")
        
        #skip idx iterations since it has already been dealt with.
        i += idx
    return predictions

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores(data, preds):
    exact_scores = []
    f1_scores = []
    for i, answers in enumerate(data['answers']):
        gold_answers = [a['text'] for a in answers if normalize_answer(a['text'])]
        if not gold_answers:
            gold_answers = ['']
        exact_scores.append(max(compute_exact(a, preds[i]) for a in gold_answers))
        f1_scores.append(max(compute_f1(a, preds[i]) for a in gold_answers))
    return exact_scores, f1_scores

def calc_param_lr(model, special_lr):
    parameters = [
        {'params':[]},
        {'params':[], 'lr':special_lr}
    ]
    for name, param in model.named_parameters():
        if 'hyper' in name:
            parameters[1]['params'].append(param)
        else:
            parameters[0]['params'].append(param)
    return parameters