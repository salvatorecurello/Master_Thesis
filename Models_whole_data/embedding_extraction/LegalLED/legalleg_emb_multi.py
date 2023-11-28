import textwrap
import keras
import random
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, LEDForSequenceClassification
import progressbar
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
import os
from sklearn.metrics import confusion_matrix

df = pd.read_csv('/home/scurello/Thesis/train_data/ILDC_multi.csv')

df_train = df.query(" split=='train' ")
df_test = df.query(" split=='test' ")
df_dev = df.query(" split=='dev' ")


def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks

def grouped_input_ids(all_toks):
  splitted_toks = []
  l=0
  r=510
  while(l<len(all_toks)):
    splitted_toks.append(all_toks[l:min(r,len(all_toks))])
    l+=410
    r+=410

  CLS = tokenizer.cls_token
  SEP = tokenizer.sep_token
  e_sents = []
  for l_t in splitted_toks:
    l_t = [CLS] + l_t + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(l_t)
    e_sents.append(encoded_sent)

  e_sents = pad_sequences(e_sents, maxlen=512, value=0, dtype="long", padding="post")
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

def input_id_maker(dataf, tokenizer):
  input_ids = []
  lengths = []

  for i in progressbar.progressbar(range(len(dataf['text']))):
    sen = dataf['text'].iloc[i] #  select document i  
    sen = tokenizer.tokenize(sen) # tokenize the document
    CLS = tokenizer.cls_token # CLS ='[CLS]'
    SEP = tokenizer.sep_token #SEP = '[SEP]'
    if(len(sen) > 510): # if the lenght of sen is > 510 then consider the last 510 tokens
      sen = sen[len(sen)-510:]

    sen = [CLS] + sen + [SEP] # add [CLS] and [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(sen) # convert the sen to ids
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

  input_ids = pad_sequences(input_ids, maxlen=510, value=0, dtype="long", truncating="post", padding="post")

  # truncating = remove values from sequences larger than maxlen, either at the beginning or at the end of the sequences.
  
  return input_ids, lengths

output_dir = '/home/scurello/Thesis/Models_whole_data/transformers/LegalLED/saved_model_multi'
device = torch.device('cuda')
model = LEDForSequenceClassification.from_pretrained(output_dir, output_hidden_states=True)
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
    vec = outputs['encoder_last_hidden_state'][0][0]
  elif strategy == 'avg':
    vec = outputs['encoder_last_hidden_state'][0].mean(dim=0)
  else:
    vec = outputs['encoder_last_hidden_state'][0].max(dim=0)[0]
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
 

# CLS

path_val_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LegalLED_multi/LegalLED_npy_files_cls_multi/LegalLED_cls_dev"
vecs_dev = generate_np_files_for_emb(df_dev, tokenizer, 'cls')
np.save(path_val_npy_file, vecs_dev)

print('npy file dev saved')

path_train_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LegalLED_multi/LegalLED_npy_files_cls_multi/LegalLED_cls_train"
vecs_train = generate_np_files_for_emb(df_train, tokenizer, 'cls')
np.save(path_train_npy_file, vecs_train)

print('npy file train saved')

path_test_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LegalLED_multi/LegalLED_npy_files_cls_multi/LegalLED_cls_test"
vecs_test = generate_np_files_for_emb(df_test, tokenizer, 'cls')
np.save(path_test_npy_file, vecs_test)

print('npy file test saved')

# AVG

path_val_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LegalLED_multi/LegalLED_npy_files_avg_multi/LegalLED_avg_dev"
vecs_dev = generate_np_files_for_emb(df_dev, tokenizer, 'avg')
np.save(path_val_npy_file, vecs_dev)

print('npy file dev saved')


path_train_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LegalLED_multi/LegalLED_npy_files_avg_multi/LegalLED_avg_train"
vecs_train = generate_np_files_for_emb(df_train, tokenizer, 'avg')
np.save(path_train_npy_file, vecs_train)

print('npy file train saved')

path_test_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LegalLED_multi/LegalLED_npy_files_avg_multi/LegalLED_avg_test"
vecs_test = generate_np_files_for_emb(df_test, tokenizer, 'avg')
np.save(path_test_npy_file, vecs_test)

print('npy file test saved')

# MAX

path_val_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LegalLED_multi/LegalLED_npy_files_max_multi/LegalLED_max_dev"
vecs_dev = generate_np_files_for_emb(df_dev, tokenizer, 'max')
np.save(path_val_npy_file, vecs_dev)

print('npy file dev saved')

path_train_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LegalLED_multi/LegalLED_npy_files_max_multi/LegalLED_max_train"
vecs_train = generate_np_files_for_emb(df_train, tokenizer, 'max')
np.save(path_train_npy_file, vecs_train)

print('npy file train saved')

path_test_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/LegalLED_multi/LegalLED_npy_files_max_multi/LegalLED_max_test"
vecs_test = generate_np_files_for_emb(df_test, tokenizer, 'max')
np.save(path_test_npy_file, vecs_test)

print('npy file test saved')