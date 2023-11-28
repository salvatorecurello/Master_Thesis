import textwrap
import keras
import random
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaForSequenceClassification
import progressbar
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
import os
from sklearn.metrics import confusion_matrix

model_type = 'RoBERTa'

'''
df = pd.read_csv('/home/scurello/Thesis/train_data/ILDC_single.csv')

df_train = df.query(" split=='train' ")
df_test = df.query(" split=='test' ")
df_dev = df.query(" split=='dev' ")
'''

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
    toks = tokenizer.tokenize(text, add_prefix_space=True)
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

'''
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
train_input_ids, train_att_masks, train_labels = generate_np_files_for_training(df_train, tokenizer)

validation_input_ids, validation_lengths = input_id_maker(df_dev, tokenizer)

validation_attention_masks = att_masking(validation_input_ids)
validation_labels = df_dev['label'].to_numpy().astype('int')

train_inputs = train_input_ids
validation_inputs = validation_input_ids
train_masks = train_att_masks
validation_masks = validation_attention_masks

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

#use batch size < 6 as it is the upper limit due to our max input length as 512 toks
batch_size = 6
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)

device = torch.device('cuda')
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model.to(device)

lr = 2e-5
max_grad_norm = 1.0
epochs = 3
num_total_steps = len(train_dataloader)*epochs
num_warmup_steps = 1000
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

seed_val = 21
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

start = time.time()

train_loss_values = []
train_accuracy = []
val_loss_values = []
val_accuracy = []

# For each epoch...
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    model.train()
    total_loss=0
    train_batch_accuracy = 0

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}. : loss: {:} '.format(step, len(train_dataloader), total_loss/step))


        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]

        total_loss+=loss.item()

        loss.backward()

        batch_logits = logits
        logits = batch_logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        train_batch_accuracy = flat_accuracy(logits, label_ids)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        if step%1000 == 0 and not step == 0:
            print("\nRunning Validation...")
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for batch in validation_dataloader:
              batch = tuple(t.to(device) for t in batch)
              b_input_ids, b_input_mask, b_labels = batch
              with torch.no_grad():        
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

              loss = outputs[0]
              logits = outputs[1]
    
              logits = logits.detach().cpu().numpy()
              label_ids = b_labels.to('cpu').numpy()
        
              tmp_eval_accuracy = flat_accuracy(logits, label_ids)
              eval_accuracy += tmp_eval_accuracy

              eval_loss+=loss

              nb_eval_steps += 1

            val_accuracy.append(eval_accuracy/nb_eval_steps)
            val_loss_values.append(eval_loss/nb_eval_steps)

            print('Validation loss: {:} : Validation accuracy: {:}'.format(val_loss_values[-1], val_accuracy[-1]))

        
    train_loss_values.append(total_loss/len(train_dataloader))
    train_accuracy.append(train_batch_accuracy/len(train_dataloader))

end = time.time()
timer(start, end, model_type, 'Training', 512)

print("")
print("Training complete!")

# Save the model
output_dir = '/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/RoBERTa_single/saved_model_single'

# Create output directory if needed
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))

# Copy the model files to a directory in your Google Drive.
#!cp -r ./BERT_final/ "/content/Drive/My Drive/BERT_right_model/"
'''

df = pd.read_csv('/home/scurello/Thesis/train_data/ILDC_multi.csv')

df_train = df.query(" split=='train' ")
df_test = df.query(" split=='test' ")
df_dev = df.query(" split=='dev' ")


output_dir = '/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/RoBERTa_single/saved_model_single'
device = torch.device('cuda')
model = RobertaForSequenceClassification.from_pretrained(output_dir, output_hidden_states=True)
tokenizer = RobertaTokenizer.from_pretrained(output_dir)
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
    vec = outputs['hidden_states'][12][0][0]
  elif strategy == 'avg':
    vec = outputs['hidden_states'][12][0].mean(dim=0)
  else:
    vec = outputs['hidden_states'][12][0].max(dim=0)[0]
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
  
path_train_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/RoBERTa_single/RoBERTa_npy_files_cls_single/RoBERTa_cls_train"
vecs_train = generate_np_files_for_emb(df_train, tokenizer, 'cls')
np.save(path_train_npy_file, vecs_train)

print('npy file train saved')

path_dev_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/RoBERTa_single/RoBERTa_npy_files_cls_single/RoBERTa_cls_dev"
vecs_dev = generate_np_files_for_emb(df_dev, tokenizer, 'cls')
np.save(path_dev_npy_file, vecs_dev)

print('npy file dev saved')

path_test_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/RoBERTa_single/RoBERTa_npy_files_cls_single/RoBERTa_cls_test"
vecs_test = generate_np_files_for_emb(df_test, tokenizer, 'cls')
np.save(path_test_npy_file, vecs_test)

print('npy file test saved')

# avg
  
path_train_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/RoBERTa_single/RoBERTa_npy_files_avg_single/RoBERTa_avg_train"
vecs_train = generate_np_files_for_emb(df_train, tokenizer, 'avg')
np.save(path_train_npy_file, vecs_train)

print('npy file train saved')

path_dev_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/RoBERTa_single/RoBERTa_npy_files_avg_single/RoBERTa_avg_dev"
vecs_dev = generate_np_files_for_emb(df_dev, tokenizer, 'avg')
np.save(path_dev_npy_file, vecs_dev)

print('npy file dev saved')

path_test_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/RoBERTa_single/RoBERTa_npy_files_avg_single/RoBERTa_avg_test"
vecs_test = generate_np_files_for_emb(df_test, tokenizer, 'avg')
np.save(path_test_npy_file, vecs_test)

print('npy file test saved')

# MAX
  
path_train_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/RoBERTa_single/RoBERTa_npy_files_max_single/RoBERTa_max_train"
vecs_train = generate_np_files_for_emb(df_train, tokenizer, 'max')
np.save(path_train_npy_file, vecs_train)

print('npy file train saved')

path_dev_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/RoBERTa_single/RoBERTa_npy_files_max_single/RoBERTa_max_dev"
vecs_dev = generate_np_files_for_emb(df_dev, tokenizer, 'max')
np.save(path_dev_npy_file, vecs_dev)

print('npy file dev saved')

path_test_npy_file = "/home/scurello/Thesis/Models_whole_data/transformers_sentence_level/RoBERTa_single/RoBERTa_npy_files_max_single/RoBERTa_max_test"
vecs_test = generate_np_files_for_emb(df_test, tokenizer, 'max')
np.save(path_test_npy_file, vecs_test)

print('npy file test saved')