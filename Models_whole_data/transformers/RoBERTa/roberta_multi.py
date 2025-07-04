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

model_type = 'RoBERTa'
n_toks = 512
df = pd.read_csv('/home/scurello/Thesis/train_data/ILDC_multi.csv')

df_train = df.query(" split=='train' ")
df_test = df.query(" split=='test' ")
df_dev = df.query(" split=='dev' ")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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

def input_id_maker(dataf, tokenizer, n_toks):
  input_ids = []
  lengths = []

  for i in progressbar.progressbar(range(len(dataf['text']))):
    sen = dataf['text'].iloc[i] #  select document i

    #------------------FUNCTION CHECK MAX SUPPORTED LENGHT----------------

    #sen_check = sen.split()
    #if len(sen_check) > 10000: # max lenght supported by this tokenizer (16384)
    #  sen = sen_check[len(sen_check)-10000:] # we consider the last 10k tokens of each document
    #  sen = " ".join(sen) # we join the last 10k tokens to retrieve the old sentence 

    #----------------------------------------------------------------------
    
    sen = tokenizer.tokenize(sen, add_prefix_space=True) # tokenize the document
    CLS = tokenizer.cls_token # CLS ='[CLS]'
    SEP = tokenizer.sep_token #SEP = '[SEP]'
    if(len(sen) > (n_toks-2)): # if the lenght of sen is > 510 then consider the last 510 tokens
      sen = sen[len(sen)-(n_toks-2):]

    sen = [CLS] + sen + [SEP] # add [CLS] and [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(sen) # convert the sen to ids
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

  input_ids = pad_sequences(input_ids, maxlen=n_toks, value=0, dtype="long", truncating="post", padding="post")

  # truncating = remove values from sequences larger than maxlen, either at the beginning or at the end of the sequences.
  
  return input_ids, lengths

train_input_ids, train_lengths = input_id_maker(df_train, tokenizer, n_toks)
validation_input_ids, validation_lengths = input_id_maker(df_dev, tokenizer, n_toks)

train_attention_masks = att_masking(train_input_ids)
validation_attention_masks = att_masking(validation_input_ids)

train_labels = df_train['label'].to_numpy().astype('int')
validation_labels = df_dev['label'].to_numpy().astype('int')

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

seed_val = 34
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

print("Training start!")

start = time.time()

loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # t0 = time.time()
    total_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))

        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        
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

    # t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
          outputs = model(b_input_ids, attention_mask=b_input_mask)
    
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

end = time.time()
timer(start, end, model_type, 'Training', n_toks)

print("")
print("Training complete!")


#save the trained model
output_dir = '/home/scurello/Thesis/Models_whole_data/transformers/RoBERTa/saved_model_multi_5ep'

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

################ PREDICTIONS ON TEST ###################

labels = df_test.label.to_numpy().astype(int)

input_ids, input_lengths = input_id_maker(df_test, tokenizer, n_toks)
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

f_accuracy = flat_accuracy(predictions,true_labels)

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

print('PREDICTIONS ON TEST SET')
print('Accuracy:')
print(f_accuracy)
macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(pred_flat, labels_flat)
print('macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1')
print(macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)


################ PREDICTIONS ON VALIDATION SET ###################
batch_size = 6  

# Create the DataLoader.
prediction_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
prediction_sampler = SequentialSampler(validation_data)
prediction_dataloader = DataLoader(validation_data, sampler=prediction_sampler, batch_size = batch_size)

print('Predicting labels for validation sentences...')
model.eval()

predictions , true_labels = [], []

for (step, batch) in enumerate(prediction_dataloader):
  batch = tuple(t.to(device) for t in batch)
  
  b_input_ids, b_input_mask, b_labels = batch

  with torch.no_grad():
      outputs = model(b_input_ids, attention_mask=b_input_mask)

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

f_accuracy = flat_accuracy(predictions,true_labels)

print('PREDICTIONS ON VALIDATION SET')
print('Accuracy:')
print(f_accuracy)
macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(pred_flat, labels_flat)
print('macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1')
print(macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)
