import pandas as pd
from transformers import BartTokenizer, BartModelWithHeads, BartConfig, TrainingArguments, AdapterTrainer, EvalPrediction, TextClassificationPipeline, set_seed
from datasets import load_dataset, load_metric
import numpy as np
from tqdm import tqdm
import random
import os
import torch

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
PATH = '/data/dataset/StandardStream/test/udpos.csv'
df_test = pd.read_csv(PATH)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
config = BartConfig.from_pretrained('facebook/bart-base', num_labels=2)
model = BartModelWithHeads.from_pretrained('facebook/bart-base', config=config)
model.load_adapter('/data/moongi/lab/BART-adapter/training_output_udpos/checkpoint-942/binary_classification')
model.set_active_adapters(['binary_classification'])

X_test = df_test['statement'] + ' ' + tokenizer.sep_token + ' ' + df_test['context']

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

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

print('Accuracy: ', count / len(preds) * 100)
