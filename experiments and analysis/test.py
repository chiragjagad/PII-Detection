"""
Credit to Valentin Werner [.915] Derbeta3base Training
"""

import json
import argparse
from itertools import chain
from functools import partial

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
import evaluate
from datasets import Dataset, features
import numpy as np

print(torch.cuda.is_available())
if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev)


PATH = ''

data = json.load(open(PATH + "kaggle/input/pii-detection-removal-from-educational-data/train.json"))

# downsampling of negative examples
p=[] # positive samples (contain relevant labels)
n=[] # negative samples (presumably contain entities that are possibly wrongly classified as entity)
for d in data:
    if any(np.array(d["labels"]) != "O"): p.append(d)
    else: n.append(d)
print("original datapoints: ", len(data))

all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i,l in enumerate(all_labels)}
id2label = {v: k for k,v in label2id.items()}

target = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM',
    'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM',
    'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL'
]

print(id2label)

def tokenize(example, tokenizer, label2id, max_length):

    # rebuild text from tokens
    text = []
    labels = []

    for t, l, ws in zip(
        example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    ):
        text.append(t)
        labels.extend([l] * len(t))

        if ws:
            text.append(" ")
            labels.append("O")

    # actual tokenization
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)

    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {**tokenized, "labels": token_labels, "length": length}

TRAIN_MODEL_PATH = "microsoft/deberta-base"
# # TRAIN_MODEL_PATH = "dslim/bert-large-NER"
# TRAIN_MODEL_PATH = "bert-base-uncased"

TRAIN_MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(TRAIN_MODEL_PATH)

ds = Dataset.from_dict({
    "full_text": [x["full_text"] for x in data],
    "document": [str(x["document"]) for x in data],
    "tokens": [x["tokens"] for x in data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    "provided_labels": [x["labels"] for x in data],
})

ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAIN_MAX_LENGTH}, num_proc=3)

x = ds[0]
i = 1
for t,l in zip(x["tokens"], x["provided_labels"]):
    if l == 'O':
        if i == 1:
          print((t,l))
          i -= 1
    if l != "O":
        print((t,l))

print("*"*100)

for t, l in zip(tokenizer.convert_ids_to_tokens(x["input_ids"]), x["labels"]):
    if id2label[l] != "O":
        print((t,id2label[l]))

ds = ds.train_test_split(test_size=0.1)

from seqeval.metrics import precision_score, recall_score

def metrics(p, all_labels):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
            [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
    true_labels = [
            [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]

    precision = precision_score(true_labels, true_predictions, average='micro')
    recall = recall_score(true_labels, true_predictions, average='micro')

    f5_score = (1 + 5 ** 2) * (precision * recall) / (5 ** 2 * precision + recall)

    results = {
        "precision": precision,
        "recall": recall,
        "f5": f5_score
    }
    print(results)
    return results

def metrics2(preds, labels):
    preds = np.argmax(preds, axis=2)
    # Remove ignored index (special tokens)
    preds = [pred[:len(label)] for pred, label in zip(preds, labels)]

    precision = precision_score(preds, labels)
    recall = recall_score(preds, labels)

    f5_score = (1 + 5 ** 2) * (precision * recall) / (5 ** 2 * precision + recall)

    results = {
        "precision": precision,
        "recall": recall,
        "f5": f5_score
    }

    return results

model_path = PATH + "deberta3base_1024_downsampled"

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.to(device)

args = TrainingArguments(
        ".",
        per_device_eval_batch_size=1,
        report_to="none",
    )

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=partial(metrics, all_labels=all_labels)
)
# # Assuming 'trainer' is your Trainer object
# if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
#     loss_history = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
#     print(loss_history)
# else:
#     print("No loss history found.")
preds,labels,metric = trainer.predict(ds['test'])

metrics((preds, labels), all_labels)
