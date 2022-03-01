from transformers import BartTokenizer, BartModelWithHeads, BartConfig, TrainingArguments, AdapterTrainer, EvalPrediction, set_seed
from datasets import load_dataset, load_metric
import numpy as np
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

TRAIN = "/data/dataset/StandardStream/train/yelp_review.csv"
train_datasets = load_dataset("csv", data_files={"train":TRAIN})

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
config = BartConfig.from_pretrained('facebook/bart-base', num_labels=2)
model = BartModelWithHeads.from_pretrained('facebook/bart-base', config=config)
model.add_adapter("binary_classification")
# Add a matching classification head
model.add_classification_head("binary_classification",num_labels=2)
# Activate the adapter
model.train_adapter('binary_classification')

def tokenize_and_encode(examples): 
    return tokenizer(examples['statement'], examples['context'], max_length=256, truncation=True, padding='max_length')

train_data = train_datasets['train'].map(tokenize_and_encode, batched=True)

def compute_metrics(p: EvalPrediction):
    metric_acc = load_metric("accuracy")    
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]

    return result

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    save_strategy='epoch',
    logging_strategy='epoch',
    per_device_train_batch_size=32,
    logging_steps=200,
    output_dir="./training_output_yelp",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
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

print('Try to save model')
model_dir = '/data/moongi/lab/models/'
trainer.save_model(model_dir + 'bart_standard_yelp/model')
