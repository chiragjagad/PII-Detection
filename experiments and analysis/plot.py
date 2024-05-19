import json
import matplotlib.pyplot as plt

PATH = "plots/"
name = "deberta"
name2 = "bert"
name3 = "distilbert"
# Step 1: Load JSON file
with open(f'training_logs_{name}_downsampling.json', 'r') as f:
    deberta_logs = json.load(f)
with open(f'training_logs_{name2}_downsampling.json', 'r') as f:
    bert_logs = json.load(f)
with open(f'training_logs_{name3}_downsampling.json', 'r') as f:
    distilbert_logs = json.load(f)
# Extract relevant information
epochs = []
train_loss = []
eval_loss = []
eval_precision = []
eval_recall = []
eval_f5 = []

epochs2 = []
train_loss2 = []
eval_loss2 = []
eval_precision2 = []
eval_recall2 = []
eval_f52 = []

epochs3 = []
train_loss3 = []
eval_loss3 = []
eval_precision3 = []
eval_recall3 = []
eval_f53 = []
train_time = 0.0
train_time2 = 0.0
train_time3 = 0.0
i = 0
for k, log in enumerate(deberta_logs):
    if i > 0:
        if epochs[i-1] != log['epoch']:
            epochs.append(log['epoch'])
            i+=1
    else: 
        epochs.append(log['epoch'])
        i += 1
    if 'loss' in log:
        train_loss.append(log['loss'])
    # elif 'train_loss' in log: 
    #     train_loss.append(log['train_loss'])        
    if 'eval_loss' in log:
        eval_loss.append(log['eval_loss'])
    if 'eval_precision' in log:
        eval_precision.append(log['eval_precision'])
    if 'eval_recall' in log:
        eval_recall.append(log['eval_recall'])
    if 'eval_f5' in log:
        eval_f5.append(log['eval_f5'])
    if 'train_runtime' in log:
        train_time = log['train_runtime']

i = 0
for k, log in enumerate(bert_logs):
    if i > 0:
        if epochs2[i-1] != log['epoch']:
            epochs2.append(log['epoch'])
            i+=1
    else: 
        epochs2.append(log['epoch'])
        i += 1
    if 'loss' in log:
        train_loss2.append(log['loss'])
    # elif 'train_loss' in log: 
    #     train_loss.append(log['train_loss'])        
    if 'eval_loss' in log:
        eval_loss2.append(log['eval_loss'])
    if 'eval_precision' in log:
        eval_precision2.append(log['eval_precision'])
    if 'eval_recall' in log:
        eval_recall2.append(log['eval_recall'])
    if 'eval_f5' in log:
        eval_f52.append(log['eval_f5'])
    if 'train_runtime' in log:
        train_time2 = log['train_runtime']

i = 0
for k, log in enumerate(distilbert_logs):
    if i > 0:
        if epochs3[i-1] != log['epoch']:
            epochs3.append(log['epoch'])
            i+=1
    else: 
        epochs3.append(log['epoch'])
        i += 1
    if 'loss' in log:
        train_loss3.append(log['loss'])
    # elif 'train_loss' in log: 
    #     train_loss.append(log['train_loss'])        
    if 'eval_loss' in log:
        eval_loss3.append(log['eval_loss'])
    if 'eval_precision' in log:
        eval_precision3.append(log['eval_precision'])
    if 'eval_recall' in log:
        eval_recall3.append(log['eval_recall'])
    if 'eval_f5' in log:
        eval_f53.append(log['eval_f5'])
    if 'train_runtime' in log:
        train_time3 =log['train_runtime']

def plot_loss_metrics(epochs = [], train_loss = [], eval_loss = [], eval_precision = [], eval_recall = [], eval_f5 = [], name = ""):
    upper_name = name.upper()
    # Create plots
    plt.figure(figsize=(10, 6))
    plt.plot(epochs[:len(train_loss)], train_loss, label='Training Loss', linewidth=2.5)
    plt.plot(epochs[:len(train_loss)], eval_loss, label='Validation Loss', linewidth=2.5)
    plt.title(f'{upper_name}: Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(PATH+f"{name}_Loss.png")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs[:len(train_loss)], eval_precision, label='Precision', color='green', linestyle='--', marker='o')
    plt.plot(epochs[:len(train_loss)], eval_recall, label='Recall', color='red', linestyle='--', marker='x')
    plt.plot(epochs[:len(train_loss)], eval_f5, label='F5', color='purple', linestyle='--', marker='s')
    plt.title(f'{upper_name}: Precision, Recall, and F5 Score', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(PATH+f"{name}_metrics.png")

# plot_loss_metrics(epochs3, train_loss3, eval_loss3, eval_precision3, eval_recall3, eval_f53, name3)

# # Compare F5 in 3 models
# def plot_F5_compare():
#     plt.figure(figsize=(10, 6))

#     plt.plot(epochs[:len(train_loss)], eval_f5, label='DeBERTa', color='red', linestyle='--', marker='o')
#     plt.plot(epochs2[:len(train_loss2)], eval_f52, label='BERT', color='purple', linestyle='--', marker='x')
#     plt.plot(epochs3[:len(train_loss3)], eval_f53, label='DistilBERT', color='purple', linestyle='--', marker='s')
#     plt.title('DeBERTa vs BERT vs DistilBERT', fontsize=16)
#     plt.xlabel('Epoch', fontsize=14)
#     plt.ylabel('F5', fontsize=14)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.savefig(PATH+"F5 comparison.png")
# plot_F5_compare()

# # Compare other metrics in 3 models
# def plot_Performance_compare():
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs[:len(train_loss)], eval_precision, label='DeBERTa (Precision)', color='purple', linestyle='-', marker='o')
#     plt.plot(epochs2[:len(train_loss2)], eval_precision2, label='BERT (Precision)', color='red', linestyle='-', marker='x')
#     plt.plot(epochs3[:len(train_loss3)], eval_precision3, label='DistilBERT (Precision)', color='purple', linestyle='-', marker='s')
#     plt.plot(epochs[:len(train_loss)], eval_recall, label='DeBERTa (Recall)', color='red', linestyle='--', marker='o')
#     plt.plot(epochs2[:len(train_loss2)], eval_recall2, label='BERT (Recall)', color='purple', linestyle='--', marker='x')
#     plt.plot(epochs3[:len(train_loss3)], eval_recall3, label='DistilBERT (Recall)', color='purple', linestyle='--', marker='s')
#     plt.title('Precision and Recall', fontsize=16)
#     plt.xlabel('Epoch', fontsize=14)
#     plt.ylabel('Precision or Recall', fontsize=14)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.savefig(PATH+"Performance comparison.png")


# # Compare training runtime
# models = ['DeBERTa', 'BERT', 'DistilBERT']
# training_times = [train_time, train_time2, train_time3]

# # Create bar chart
# plt.figure(figsize=(5, 5))
# plt.bar(models, training_times, color=['blue', 'blue', 'red'], width=0.5)
# plt.title('Training Runtime Comparison', fontsize=16)
# plt.xlabel('Models', fontsize=14)
# plt.ylabel('Training Time (seconds)', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.tight_layout()
# plt.savefig(PATH+"Runtime Efficiency.png")