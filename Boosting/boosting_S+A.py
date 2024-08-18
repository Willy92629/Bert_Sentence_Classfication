# %%
import json
with open("train_social_att_sup.json",'r',encoding='utf-8')as f:
    train_data=json.load(f)
with open("dev_social_att_sup.json",'r',encoding='utf-8')as f:
    eval_data=json.load(f)

# %%
weight=[1/len(train_data) for x in range(len(train_data))]

# %%
def read_data(dataset):
    sentence1=[]
    sentence2=[]
    label=[]
    idx=[]
    for text in dataset:
        sentence1.append(text[0])
        sentence2.append(text[1])
        label.append(text[2])
    return sentence1,sentence2,label

# %%
ori_train_sen1,ori_train_sen2,ori_trainlabel=read_data(train_data)
print(len(ori_train_sen1))
print(ori_train_sen1[0])
eval_sen1,eval_sen2,evallabel=read_data(eval_data)
print(len(eval_sen1))
print(eval_sen1[0])
print(ori_trainlabel)

# %%
print('train 資料集數量= ',len(train_data))
print('eval 資料集數量= ',len(eval_data))

# %%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# %%
import transformers
transformers.logging.set_verbosity_error()

# %% [markdown]
# # 主要修改以下的東西

# %%
total_epoch=5
count_epoch=0
best_epoch = {"epoch": 0, "acc": 0 }
all_epoch=[]

# %%
from torch.utils import data
import torch
def add_targets(encodings,label):
    encodings.update({'label':label})

class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __getitem__(self, idx):
    return {key: torch.tensor(eval[idx]) for key, eval in self.encodings.items()}

  def __len__(self):
    return len(self.encodings.input_ids)

# %%
def eval_model(model, sen1, sen2):
  input_encodings = tokenizer([sen1], [sen2], padding='max_length', truncation=True)
  input_dataset = Dataset(input_encodings)
  data_collator = default_data_collator
  input_dataloader = DataLoader(input_dataset, collate_fn=data_collator, batch_size=1)

  accelerator = Accelerator()
  model, input_dataloader = accelerator.prepare(model, input_dataloader)

  for batch in input_dataloader:
    outputs = model(**batch)
    predicted = outputs.logits.argmax(dim=-1)
  return predicted

# %%
eval_encodings = tokenizer(eval_sen1, eval_sen2, truncation=True, padding=True)
add_targets(eval_encodings,evallabel)
eval_dataset = Dataset(eval_encodings)

# %%
import numpy as np
import math
import logging
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import (
    AdamW,
    default_data_collator,
    get_scheduler,
    BertConfig,
    BertForSequenceClassification
)
from accelerate import Accelerator

alpha_list = [] #投票決定時需要alpha
all_epoch = []  # List to store epoch accuracy results
best_epoch = {'epoch': -1, 'acc': 0.0}  # Dictionary to store the best epoch
train_batch_size = 4      # 設定 training batch size 
eval_batch_size = 4      # 設定 eval batch size
num_train_epochs = 5    #設定單模型 epoch
data_collator = default_data_collator
learning_rate=3e-5          # 設定 learning_rate
gradient_accumulation_steps = 1   # 設定 幾步後進行反向傳播
no_decay = ["bias", "LayerNorm.weight"]
output_dir = '.'

for count_epoch in range(total_epoch):
    print("epoch: ",count_epoch)
    train_sen1, train_sen2, trainlabel = read_data(train_data)
    train_encodings = tokenizer(train_sen1, train_sen2, truncation=True, padding=True)
    add_targets(train_encodings, trainlabel)
    
    train_dataset = Dataset(train_encodings)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=eval_batch_size)

    # Load or initialize the model
    if count_epoch == 0:
        config = BertConfig.from_pretrained('bert-base-chinese', num_labels=2)
        model = BertForSequenceClassification.from_pretrained("bert-base-chinese", config=config)
    
    # Initialize optimizer and scheduler
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )
    
    accelerator = Accelerator()
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    completed_steps = 0
    for epoch in trange(num_train_epochs, desc="Epoch"):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            if step % 200 == 0:
                print({'epoch': epoch, 'step': step, 'loss': loss.item()})
            if completed_steps >= max_train_steps:
                break

        # Evaluation
        logger.info("***** Running eval *****")
        model.eval()
        metric = load_metric("accuracy")
        for step, batch in enumerate(tqdm(eval_dataloader, desc="eval Iteration")):
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in batch.items()}  
                outputs = model(**inputs)
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")
        print("eval accuracy: ", eval_metric['accuracy'])
        all_epoch.append([epoch, eval_metric['accuracy']])
        if eval_metric['accuracy'] > best_epoch['acc']:
            best_epoch.update({"epoch": epoch, "acc": eval_metric['accuracy']})

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir + '/' + 'epoch_' + str(count_epoch), save_function=accelerator.save)

    cnt = 0
    correct_arr = []
    errorcnt = 0
    for i in range(len(train_data)):
        cnt += 1
        sen1 = ori_train_sen1[i]
        sen2 = ori_train_sen2[i]
        predict = eval_model(model, sen1, sen2)  # Need to define eval_model() correctly

        if predict.item() != ori_trainlabel[i]:
            correct_arr.append(-1)
            errorcnt += 1
        else:
            correct_arr.append(1)

    accuracy = (cnt - errorcnt)/cnt
    error_rate = errorcnt / cnt
    alpha = 0.5 * (np.log((1 - error_rate) / error_rate))
    alpha_list.append(alpha)
    #weight *= np.exp(-alpha * trainlabel * correct_arr)
    for i in range(len(weight)):           
        weight[i] *= np.exp(-alpha * correct_arr[i])
    weightdiv = np.sum(weight)  # Normalize the weights
    for i in range(len(weight)):           
        weight[i] /= weightdiv

    print("accuracy: ",accuracy)
    print("alpha: ",alpha)

# %%
print(alpha_list)

# %%
import numpy as np

def adaboost_voting(model_amount, models, alpha, sen1, sen2):
    votes = 0
    for i in range(model_amount):
        model = models[i]
        model_vote = eval_model(model, sen1, sen2).item()
        votes += model_vote * alpha[i]
    votes /= np.sum(alpha) #調整在0~1之間
    final_prediction = 1 if votes > 0.5 else 0 #靠近0就是預測0，靠近1就是預測1
    return final_prediction

# %%
from transformers import BertConfig, BertForSequenceClassification
models = [] #存多個模型
model_amount = 5 #可以透過這個參數決定要用幾個model預測
for i in range(model_amount):
    config = BertConfig.from_pretrained(f'./epoch_{i}/config.json')
    model = BertForSequenceClassification.from_pretrained(f'./epoch_{i}/pytorch_model.bin', config=config)
    models.append(model)

# %%
import numpy as np
cnt=0
correct_arr=[]
errorcnt=0
for i in tqdm(range(len(eval_data))):
    cnt+=1
    sen1=eval_sen1[i]
    sen2=eval_sen2[i]
    predict=adaboost_voting(model_amount, models,alpha_list,sen1,sen2)

    if predict!=evallabel[i]:
        correct_arr.append(0)
        errorcnt+=1
    else:
        correct_arr.append(1)


accuracy=(cnt-errorcnt)/cnt
error_rate = errorcnt/cnt
alpha=0.5*(np.log((1-error_rate)/error_rate))

print(f'cnt = {cnt},errorcnt = {errorcnt}')
print("accuracy: "+str(accuracy))
print("alpha: "+str(alpha))

# %%
print(f'cnt = {cnt},errorcnt = {errorcnt}')
print("accuracy: "+str(accuracy))
print("alpha: "+str(alpha))

# %%
import numpy as np
model_i=0
for model in models:
    cnt=0
    correct_arr=[]
    errorcnt=0
    for i in tqdm(range(len(eval_data))):
        cnt+=1
        sen1=eval_sen1[i]
        sen2=eval_sen2[i]
        predict=eval_model(model,sen1,sen2)
        if predict.item()!=evallabel[i]:
            correct_arr.append(0)
            errorcnt+=1
        else:
            correct_arr.append(1)
    accuracy=(cnt-errorcnt)/cnt
    error_rate = errorcnt/cnt
    alpha=0.5*(np.log((1-error_rate)/error_rate))
    print(f"model {model_i}: ")
    print(f'cnt = {cnt},errorcnt = {errorcnt}')
    print("accuracy: "+str(accuracy))
    print("alpha: "+str(alpha))
    model_i+=1

# %%
print(best_epoch)

# %%
print(all_epoch)


