import pandas as pd
from transformers import BartTokenizer, BartModelWithHeads, BartConfig, TrainingArguments, AdapterTrainer, EvalPrediction, TextClassificationPipeline, set_seed
from datasets import load_dataset, load_metric
import numpy as np
import torch
import random
import os
from tqdm import tqdm

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
acc = {}

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
config = BartConfig.from_pretrained('facebook/bart-base', num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenize_and_encode(examples): 
    return tokenizer(examples['statement'], examples['context'], max_length=256, truncation=True, padding='max_length')

def compute_metrics(p: EvalPrediction):
    metric_acc = load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]

    return result

#BoolQ TRAIN
TRAIN = '/tmp/dataset/StandardStream/train/boolq.csv'
train_datasets = load_dataset("csv", data_files={"train":TRAIN})

model = BartModelWithHeads.from_pretrained('facebook/bart-base', config=config).to(device)
model.add_adapter("boolq")
model.add_classification_head("binary_classification", num_labels=2)
model.train_adapter('boolq')

train_data = train_datasets['train'].map(tokenize_and_encode, batched=True)

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    save_strategy='epoch',
    logging_strategy='epoch',
    per_device_train_batch_size=32,
    logging_steps=200,
    output_dir="./sequence_training_output_boolq",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    dataloader_num_workers=1
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    compute_metrics=compute_metrics,
)

trainer.train()
#BoolQ Test
TEST = '/tmp/dataset/StandardStream/test/boolq.csv'
df_test = pd.read_csv(TEST)
X_test = df_test['statement'] + ' ' + tokenizer.sep_token + ' ' + df_test['context']

classifier = TextClassificationPipeline(model= model, tokenizer= tokenizer, device = 0)

preds = []
for i in tqdm(range(len(X_test))):
    pred = classifier(X_test[i])[0]['label']
    if pred == 'LABEL_0':
        preds.append(0)
    elif pred == 'LABEL_1':
        preds.append(1)
    
real = list(df_test['label'])

count = 0
for i in range(len(preds)):
    if preds[i] == real[i]:
        count += 1
accuracy = count / len(preds) * 100
print('BoolQ Accuracy: ', accuracy)
acc['BoolQ'] = accuracy

#UDPOS TRAIN
TRAIN = '/tmp/dataset/StandardStream/train/udpos.csv'
train_datasets = load_dataset("csv", data_files={"train":TRAIN})
train_data = train_datasets['train'].map(tokenize_and_encode, batched=True)

model.add_adapter('udpos')
model.train_adapter('udpos')

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    save_strategy='epoch',
    logging_strategy='epoch',
    per_device_train_batch_size=32,
    logging_steps=200,
    output_dir="./sequence_training_output_udpos",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    dataloader_num_workers=1
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    compute_metrics=compute_metrics,
)

trainer.train()

#UDPOS Test
TEST = '/tmp/dataset/StandardStream/test/udpos.csv'
df_test = pd.read_csv(TEST)
X_test = df_test['statement'] + ' ' + tokenizer.sep_token + ' ' + df_test['context']

classifier = TextClassificationPipeline(model= model, tokenizer= tokenizer, device = 0)

preds = []
for i in tqdm(range(len(X_test))):
    pred = classifier(X_test[i])[0]['label']
    if pred == 'LABEL_0':
        preds.append(0)
    elif pred == 'LABEL_1':
        preds.append(1)
    
real = list(df_test['label'])

count = 0
for i in range(len(preds)):
    if preds[i] == real[i]:
        count += 1
accuracy = count / len(preds) * 100
print('UDPOS Accuracy: ', accuracy)
acc['UDPOS'] = accuracy

#WiC Train
TRAIN = '/tmp/dataset/StandardStream/train/wic.csv'
train_datasets = load_dataset("csv", data_files={"train":TRAIN})
train_data = train_datasets['train'].map(tokenize_and_encode, batched=True)

model.add_adapter('wic')
model.train_adapter('wic')

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    save_strategy='epoch',
    logging_strategy='epoch',
    per_device_train_batch_size=32,
    logging_steps=200,
    output_dir="./sequence_training_output_wic",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    dataloader_num_workers=1
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    compute_metrics=compute_metrics,
)

trainer.train()

#WiC Test
TEST = '/tmp/dataset/StandardStream/test/wic.csv'
df_test = pd.read_csv(TEST)
X_test = df_test['statement'] + ' ' + tokenizer.sep_token + ' ' + df_test['context']

classifier = TextClassificationPipeline(model= model, tokenizer= tokenizer, device = 0)

preds = []
for i in tqdm(range(len(X_test))):
    pred = classifier(X_test[i])[0]['label']
    if pred == 'LABEL_0':
        preds.append(0)
    elif pred == 'LABEL_1':
        preds.append(1)
    
real = list(df_test['label'])

count = 0
for i in range(len(preds)):
    if preds[i] == real[i]:
        count += 1
accuracy = count / len(preds) * 100
print('WiC Accuracy: ', accuracy)
acc['WiC'] = accuracy

#FewRel Train
TRAIN = '/tmp/dataset/StandardStream/train/few_rel.csv'
train_datasets = load_dataset("csv", data_files={"train":TRAIN})
train_dataset = train_datasets['train'].map(tokenize_and_encode, batched=True)

model.add_adapter('fewrel')
model.train_adapter('fewrel')

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    save_strategy='epoch',
    logging_strategy='epoch',
    per_device_train_batch_size=32,
    logging_steps=200,
    output_dir="./sequence_training_output_fewrel",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    dataloader_num_workers=1
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    compute_metrics=compute_metrics,
)

trainer.train()

#FewRel Test
TEST = '/tmp/dataset/StandardStream/test/few_rel.csv'
df_test = pd.read_csv(TEST)
X_test = df_test['statement'] + ' ' + tokenizer.sep_token + ' ' + df_test['context']

classifier = TextClassificationPipeline(model= model, tokenizer= tokenizer, device = 0)

preds = []
for i in tqdm(range(len(X_test))):
    pred = classifier(X_test[i])[0]['label']
    if pred == 'LABEL_0':
        preds.append(0)
    elif pred == 'LABEL_1':
        preds.append(1)
    
real = list(df_test['label'])

count = 0
for i in range(len(preds)):
    if preds[i] == real[i]:
        count += 1
accuracy = count / len(preds) * 100
print('FewRel Accuracy: ', accuracy)
acc['FewRel'] = accuracy

#Yelp Train
TRAIN = '/tmp/dataset/StandardStream/train/yelp_review.csv'
train_datasets = load_dataset("csv", data_files={"train":TRAIN})
train_dataset = train_datasets['train'].map(tokenize_and_encode, batched=True)

model.add_adapter('yelp')
model.train_adapter('yelp')

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    save_strategy='epoch',
    logging_strategy='epoch',
    per_device_train_batch_size=32,
    logging_steps=200,
    output_dir="./sequence_training_output_yelp",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    dataloader_num_workers=1
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    compute_metrics=compute_metrics,
)

trainer.train()

adapter_dir = '/data/moongi/lab/adapter/BART/'
trainer.save_model(adapter_dir + 'standard_sequence/final/')
print(acc)

#Yelp Test
TEST = '/tmp/dataset/StandardStream/test/yelp_review.csv'
df_test = pd.read_csv(TEST)
X_test = df_test['statement'] + ' ' + tokenizer.sep_token + ' ' + df_test['context']

classifier = TextClassificationPipeline(model= model, tokenizer= tokenizer, device = 0)

preds = []
for i in tqdm(range(len(X_test))):
    pred = classifier(X_test[i])[0]['label']
    if pred == 'LABEL_0':
        preds.append(0)
    elif pred == 'LABEL_1':
        preds.append(1)
    
real = list(df_test['label'])

count = 0
for i in range(len(preds)):
    if preds[i] == real[i]:
        count += 1
accuracy = count / len(preds) * 100
print('Yelp Accuracy: ', accuracy)
acc['Yelp'] = accuracy

print(acc)
