import io
import os
import torch
import pandas as pd
import random
import numpy as np

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import set_seed, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, GPT2ForSequenceClassification

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(2022)
set_seed(2022)
max_length = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
learning_rate = 1e-4
num_train_epochs = 6
model_ckpt= 'gpt2'
n_labels = 2
labels_ids = {0: 0, 1: 1}
accuracy = {}

class StreamDataset(Dataset):
  def __init__(self, path, use_tokenizer):
    self.texts = []
    self.labels = []
    df = pd.read_csv(path)

    df['text'] = df['context'] + ' ' + use_tokenizer.sep_token + ' ' + df['statement']

    for i in range(len(df)):
        self.texts.append(df['text'][i])
        self.labels.append(df['label'][i])

    self.n_examples = len(self.labels)

    return

  def __len__(self):
    return self.n_examples

  def __getitem__(self, item):
    return {'text':self.texts[item], 'label':self.labels[item]}

class Gpt2ClassificationCollator(object):
    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]

        labels = [sequence['label'] for sequence in sequences]
        labels = [self.labels_encoder[label] for label in labels]

        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        inputs.update({'labels':torch.tensor(labels)})

        return inputs

def train(dataloader, optimizer, scheduler, device_):
  global model

  predictions_labels = []
  true_labels = []
  total_loss = 0

  model.train()

  for batch in tqdm(dataloader, total=len(dataloader)):
    true_labels += batch['labels'].numpy().flatten().tolist()

    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

    model.zero_grad()

    outputs = model(**batch)
    loss, logits = outputs[:2]
    total_loss += loss.item()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    logits = logits.detach().cpu().numpy()

    predictions_labels += logits.argmax(axis=-1).flatten().tolist()

  avg_epoch_loss = total_loss / len(dataloader)

  return true_labels, predictions_labels, avg_epoch_loss

def validation(dataloader, device_):
  global model

  predictions_labels = []
  true_labels = []
  total_loss = 0

  model.eval()

  for batch in tqdm(dataloader, total=len(dataloader)):

    true_labels += batch['labels'].numpy().flatten().tolist()

    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

    with torch.no_grad():        
        outputs = model(**batch)

        loss, logits = outputs[:2]

        logits = logits.detach().cpu().numpy()

        total_loss += loss.item()

        predict_content = logits.argmax(axis=-1).flatten().tolist()
        predictions_labels += predict_content

  avg_epoch_loss = total_loss / len(dataloader)

  return true_labels, predictions_labels, avg_epoch_loss

model_config = GPT2Config.from_pretrained(model_ckpt, num_labels=n_labels)

tokenizer = GPT2Tokenizer.from_pretrained(model_ckpt)
tokenizer.padding_side = "left"
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})

model = GPT2ForSequenceClassification.from_pretrained(model_ckpt, config=model_config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.to(device)


gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, labels_encoder=labels_ids, max_sequence_len=max_length)

#BoolQ
train_dataset = StreamDataset(path='/data/dataset/StandardStream/train/boolq.csv', use_tokenizer=tokenizer)
trainset, validset = torch.utils.data.random_split(train_dataset, [9000, 1000])

train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)

optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
total_steps = len(train_dataloader) * num_train_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

for epoch in tqdm(range(num_train_epochs)):
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
  train_acc = accuracy_score(train_labels, train_predict)

  valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
  val_acc = accuracy_score(valid_labels, valid_predict)

  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
  print()

test_dataset = StreamDataset(path='/data/dataset/StandardStream/test/boolq.csv', use_tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator) 

test_labels, test_predict, test_loss = validation(test_dataloader, device)
test_acc = accuracy_score(test_labels, test_predict)
print('test_loss: %.5f, test_acc: %.5f'%(test_loss, test_acc))
accuracy['boolq'] = test_acc

#UDPOS
train_dataset = StreamDataset(path='/data/dataset/StandardStream/train/udpos.csv', use_tokenizer=tokenizer)
trainset, validset = torch.utils.data.random_split(train_dataset, [9000, 1000])

train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)

for epoch in tqdm(range(num_train_epochs)):
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
  train_acc = accuracy_score(train_labels, train_predict)

  valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
  val_acc = accuracy_score(valid_labels, valid_predict)

  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
  print()

test_dataset = StreamDataset(path='/data/dataset/StandardStream/test/udpos.csv', use_tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)


test_labels, test_predict, test_loss = validation(test_dataloader, device)
test_acc = accuracy_score(test_labels, test_predict)
print('test_loss: %.5f, test_acc: %.5f'%(test_loss, test_acc))
accuracy['udpos'] = test_acc

#WiC
train_dataset = StreamDataset(path='/data/dataset/StandardStream/train/wic.csv', use_tokenizer=tokenizer)
trainset, validset = torch.utils.data.random_split(train_dataset, [9000, 1000])

train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)

for epoch in tqdm(range(num_train_epochs)):
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
  train_acc = accuracy_score(train_labels, train_predict)

  valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
  val_acc = accuracy_score(valid_labels, valid_predict)

  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
  print()

test_dataset = StreamDataset(path='/data/dataset/StandardStream/test/wic.csv', use_tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)

test_labels, test_predict, test_loss = validation(test_dataloader, device)
test_acc = accuracy_score(test_labels, test_predict)
print('test_loss: %.5f, test_acc: %.5f'%(test_loss, test_acc))
accuracy['wic'] = test_acc

#FewRel
train_dataset = StreamDataset(path='/data/dataset/StandardStream/train/few_rel.csv', use_tokenizer=tokenizer)
trainset, validset = torch.utils.data.random_split(train_dataset, [9000, 1000])

train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)

for epoch in tqdm(range(num_train_epochs)):
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
  train_acc = accuracy_score(train_labels, train_predict)

  valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
  val_acc = accuracy_score(valid_labels, valid_predict)

  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
  print()

test_dataset = StreamDataset(path='/data/dataset/StandardStream/test/few_rel.csv', use_tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)

test_labels, test_predict, test_loss = validation(test_dataloader, device)
test_acc = accuracy_score(test_labels, test_predict)
print('test_loss: %.5f, test_acc: %.5f'%(test_loss, test_acc))
accuracy['few_rel'] = test_acc

#Yelp
train_dataset = StreamDataset(path='/data/dataset/StandardStream/train/yelp_review.csv', use_tokenizer=tokenizer)
trainset, validset = torch.utils.data.random_split(train_dataset, [9000, 1000])

train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)

for epoch in tqdm(range(num_train_epochs)):
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
  train_acc = accuracy_score(train_labels, train_predict)

  valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
  val_acc = accuracy_score(valid_labels, valid_predict)

  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
  print()

test_dataset = StreamDataset(path='/data/dataset/StandardStream/test/yelp_review.csv', use_tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)

test_labels, test_predict, test_loss = validation(test_dataloader, device)
test_acc = accuracy_score(test_labels, test_predict)
print('test_loss: %.5f, test_acc: %.5f'%(test_loss, test_acc))
accuracy['yelp'] = test_acc

print(accuracy)
