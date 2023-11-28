import textwrap
import keras
import random
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaForSequenceClassification
import progressbar
from keras_preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
import os
from sklearn.metrics import confusion_matrix

import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

model_type = 'RoBERTa'
n_toks = 512
df = pd.read_csv('/home/scurello/Thesis/train_data/ILDC_multi.csv')

train_set = df.query(" split=='train' ")
test_set = df.query(" split=='test' ")
validation_set = df.query(" split=='dev' ")

# load all models and select roberta
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig

MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)}

model_type = 'roberta' ###--> CHANGE WHAT MODEL YOU WANT HERE!!! <--###
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
model_name = 'roberta-base'

tokenizer = tokenizer_class.from_pretrained(model_name)

def input_id_maker(dataf, tokenizer):
  input_ids = []
  lengths = []

  for i in progressbar.progressbar(range(len(dataf['text']))):
    sen = dataf['text'].iloc[i]
    sen = tokenizer.tokenize(sen, add_prefix_space=True)
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    if(len(sen) > 510):
      sen = sen[len(sen)-510:]

    sen = [CLS] + sen + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(sen)
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

  input_ids = pad_sequences(input_ids, maxlen=512, value=0, dtype="long", truncating="pre", padding="post")
  return input_ids, lengths
train_input_ids, train_lengths = input_id_maker(train_set, tokenizer)
validation_input_ids, validation_lengths = input_id_maker(validation_set, tokenizer)
def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks
train_attention_masks = att_masking(train_input_ids)
validation_attention_masks = att_masking(validation_input_ids)

train_labels = train_set['label'].to_numpy().astype('int')
validation_labels = validation_set['label'].to_numpy().astype('int')
train_inputs = train_input_ids
validation_inputs = validation_input_ids
train_masks = train_attention_masks
validation_masks = validation_attention_masks

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)
# max batch size should be 6 due to colab limits
batch_size = 6
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model.to(device)
lr = 2e-6
max_grad_norm = 1.0
epochs = 5
num_total_steps = len(train_dataloader)*epochs
num_warmup_steps = 1000
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

seed_val = 2212


np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))

        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
          outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

print("")
print("Training complete!")
#save the trained model
output_dir = '/home/scurello/Thesis/Models_whole_data/tramformers/RoBERTa/saved_model_multi'

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
#!cp -r ./RoBERTa_final2/ "/content/Drive/My Drive/RoBERTa_right_model2/"
labels = test_set.label.to_numpy().astype(int)

input_ids, input_lengths = input_id_maker(test_set, tokenizer)
attention_masks = att_masking(input_ids)

# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)

# Set the batch size.  
batch_size = 6  

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
model.eval()

predictions , true_labels = [], []

for (step, batch) in enumerate(prediction_dataloader):
  batch = tuple(t.to(device) for t in batch)
  b_input_ids, b_input_mask, b_labels = batch
  
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')
predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)
pred_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = true_labels.flatten()

acc = flat_accuracy(predictions,true_labels)
print('accuracy')
print(acc)

def metrics_calculator(preds, test_labels):
    cm = confusion_matrix(test_labels, preds)
    TP = []
    FP = []
    FN = []
    for i in range(0,2):
        summ = 0
        for j in range(0,2):
            if(i!=j):
                summ=summ+cm[i][j]

        FN.append(summ)
    for i in range(0,2):
        summ = 0
        for j in range(0,2):
            if(i!=j):
                summ=summ+cm[j][i]

        FP.append(summ)
    for i in range(0,2):
        TP.append(cm[i][i])
    precision = []
    recall = []
    for i in range(0,2):
        precision.append(TP[i]/(TP[i] + FP[i]))
        recall.append(TP[i]/(TP[i] + FN[i]))

    macro_precision = sum(precision)/2
    macro_recall = sum(recall)/2
    micro_precision = sum(TP)/(sum(TP) + sum(FP))
    micro_recall = sum(TP)/(sum(TP) + sum(FN))
    micro_f1 = (2*micro_precision*micro_recall)/(micro_precision + micro_recall)
    macro_f1 = (2*macro_precision*macro_recall)/(macro_precision + macro_recall)
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(pred_flat, labels_flat)
print(macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)
batch_size = 6  

# Create the DataLoader.
prediction_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
prediction_sampler = SequentialSampler(validation_data)
prediction_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)
print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
model.eval()

predictions , true_labels = [], []

for (step, batch) in enumerate(prediction_dataloader):
  batch = tuple(t.to(device) for t in batch)
  
  b_input_ids, b_input_mask, b_labels = batch

  with torch.no_grad():
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')
predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)
pred_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = true_labels.flatten()


acc_test = flat_accuracy(predictions,true_labels)
print('accuracy')
print(acc_test)
macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(pred_flat, labels_flat)
print(macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)