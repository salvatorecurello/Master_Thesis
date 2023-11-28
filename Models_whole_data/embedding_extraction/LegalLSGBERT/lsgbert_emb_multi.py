import textwrap
import keras
import random
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
import progressbar
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
import os
from sklearn.metrics import confusion_matrix

n_toks = 2560
df = pd.read_csv('/home/scurello/Thesis/train_data/ILDC_multi.csv')

df_train = df.query(" split=='train' ")
df_test = df.query(" split=='test' ")
df_dev = df.query(" split=='dev' ")

def timer(start,end, model, phase, n_toks):
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
  time_model = '\n'+model+' '+phase+' phase(hh:mm:ss) with '+str(n_toks)+' tokens: '+time
  print(time_model)
  #with open('/content/drive/MyDrive/Thesis/time/time_annotation.txt', 'a') as f:
  #    f.writelines(time_model)
  #    f.close()

# input_ids -> e_sents

def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks

def grouped_input_ids(all_toks):
  splitted_toks = []
  l=0
  r=2558
  while(l<len(all_toks)):
    splitted_toks.append(all_toks[l:min(r,len(all_toks))])
    l+=2458
    r+=2458

  CLS = tokenizer.cls_token
  SEP = tokenizer.sep_token
  e_sents = []
  for l_t in splitted_toks:
    l_t = [CLS] + l_t + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(l_t)
    e_sents.append(encoded_sent)

  e_sents = pad_sequences(e_sents, maxlen=2560, value=0, dtype="long", padding="post")
  att_masks = att_masking(e_sents)
  return e_sents, att_masks

def generate_np_files_for_training(dataf, tokenizer):
  all_input_ids, all_att_masks, all_labels = [], [], []
  for i in progressbar.progressbar(range(len(dataf['text']))):
    text = dataf['text'].iloc[i]
    toks = tokenizer.tokenize(text)
    if(len(toks) > 10000):
      toks = toks[len(toks)-10000:]

    splitted_input_ids, splitted_att_masks = grouped_input_ids(toks)
    doc_label = dataf['label'].iloc[i]
    for i in range(len(splitted_input_ids)):
      all_input_ids.append(splitted_input_ids[i])
      all_att_masks.append(splitted_att_masks[i])
      all_labels.append(doc_label)

  return all_input_ids, all_att_masks, all_labels


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


output_dir = '/home/scurello/Thesis/Models_whole_data/transformers/LSGBERT/saved_model_multi_2560'
device = torch.device('cuda')
model = AutoModelForSequenceClassification.from_pretrained(output_dir, output_hidden_states=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model.to(device)

def get_output_for_one_vec(input_id, att_mask, strategy):
  input_ids = torch.tensor(input_id)
  att_masks = torch.tensor(att_mask)
  input_ids = input_ids.unsqueeze(0)
  att_masks = att_masks.unsqueeze(0)
  model.eval()
  input_ids = input_ids.to(device)
  att_masks = att_masks.to(device)
  with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=att_masks)
  
  if strategy == 'cls':
    vec = outputs["hidden_states"][12][0][0]
  elif strategy == 'avg':
    vec = outputs["hidden_states"][12][0].mean(dim=0)
  else:
    vec = outputs["hidden_states"][12][0].max(dim=0)[0]
  vec = vec.detach().cpu().numpy() 
  return vec
  

def generate_np_files_for_emb(dataf, tokenizer, strategy):
  all_docs = []
  for i in progressbar.progressbar(range(len(dataf['text']))):
    text = dataf['text'].iloc[i]
    toks = tokenizer.tokenize(text)
    if(len(toks) > 10000):
      toks = toks[len(toks)-10000:]

    splitted_input_ids, splitted_att_masks = grouped_input_ids(toks)

    vecs = []
    for index,ii in enumerate(splitted_input_ids):
      vecs.append(get_output_for_one_vec(ii, splitted_att_masks[index], strategy))
  
    one_doc = np.asarray(vecs)
    all_docs.append(one_doc)
  
  all_docs = np.asarray(all_docs) 
  return all_docs 

'''
# CLS

path_val_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LSGBERT_multi/LSGBERT_npy_files_cls_multi/LSGBERT_cls_dev"
vecs_dev = generate_np_files_for_emb(df_dev, tokenizer, 'cls')
np.save(path_val_npy_file, vecs_dev)

print('npy file dev saved')

path_train_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LSGBERT_multi/LSGBERT_npy_files_cls_multi/LSGBERT_cls_train"
vecs_train = generate_np_files_for_emb(df_train, tokenizer, 'cls')
np.save(path_train_npy_file, vecs_train)

print('npy file train saved')

path_test_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LSGBERT_multi/LSGBERT_npy_files_cls_multi/LSGBERT_cls_test"
vecs_test = generate_np_files_for_emb(df_test, tokenizer, 'cls')
np.save(path_test_npy_file, vecs_test)

print('npy file test saved')

'''

# AVG

path_val_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LSGBERT_multi/LSGBERT_npy_files_avg_multi/LSGBERT_avg_dev"
vecs_dev = generate_np_files_for_emb(df_dev, tokenizer, 'avg')
np.save(path_val_npy_file, vecs_dev)

print('npy file dev saved')

path_train_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LSGBERT_multi/LSGBERT_npy_files_avg_multi/LSGBERT_avg_train"
vecs_train = generate_np_files_for_emb(df_train, tokenizer, 'avg')
np.save(path_train_npy_file, vecs_train)

print('npy file train saved')

path_test_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LSGBERT_multi/LSGBERT_npy_files_avg_multi/LSGBERT_avg_test"
vecs_test = generate_np_files_for_emb(df_test, tokenizer, 'avg')
np.save(path_test_npy_file, vecs_test)

print('npy file test saved')

# MAX

path_val_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LSGBERT_multi/LSGBERT_npy_files_max_multi/LSGBERT_max_dev"
vecs_dev = generate_np_files_for_emb(df_dev, tokenizer, 'max')
np.save(path_val_npy_file, vecs_dev)

print('npy file dev saved')

path_train_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LSGBERT_multi/LSGBERT_npy_files_max_multi/LSGBERT_max_train"
vecs_train = generate_np_files_for_emb(df_train, tokenizer, 'max')
np.save(path_train_npy_file, vecs_train)

print('npy file train saved')

path_test_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LSGBERT_multi/LSGBERT_npy_files_max_multi/LSGBERT_max_test"
vecs_test = generate_np_files_for_emb(df_test, tokenizer, 'max')
np.save(path_test_npy_file, vecs_test)

print('npy file test saved')
