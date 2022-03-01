import io
import os
import torch
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import set_seed, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, GPT2ForSequenceClassification

set_seed(2022)
max_length = 256
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ckpt= 'gpt2'
n_labels = 2
labels_ids = {0: 0, 1: 1}

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

def test(dataloader, device_):
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

PATH = '/data/moongi/lab/GPT_fixed/udpos'
print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(PATH, num_labels=n_labels)

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(model_ckpt)
tokenizer.padding_side = "left"
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})

print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(PATH, config=model_config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.to(device)
print('Model loaded to `%s`'%device)

gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, labels_encoder=labels_ids, max_sequence_len=max_length)

print('Creating datasets...')
test_dataset = StreamDataset(path='/data/dataset/StandardStream/test/udpos.csv', use_tokenizer=tokenizer)
print('Created `test_dataset` with %d examples!'%len(test_dataset))

print('Creating dataloaders...')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
print('Created `test_dataloader` with %d batches!'%len(test_dataloader))
print()

test_labels, test_predict, test_loss = test(test_dataloader, device)
test_acc = accuracy_score(test_labels, test_predict)
print('test_loss: %.5f, test_acc: %.5f'%(test_loss, test_acc))
