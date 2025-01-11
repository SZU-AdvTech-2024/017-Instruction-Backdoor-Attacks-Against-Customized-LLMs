import huggingface_hub    # 导入huggingface_hub库，用于访问Hugging Face模型和数据集
from transformers import AutoTokenizer, AutoConfig
import transformers
import torch
import os
from tqdm import tqdm
import datasets
from datasets import load_dataset
# import datasets
from torch.utils.data import DataLoader
import sys
import numpy as np
import argparse
import os
from utils.instructions import instructions
import requests  # 导入requests库，用于发送HTTP请求
import json 

# 创建ArgumentParser对象，用于解析命令行参数
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--model", type=str, default="GPT3.5", help='choose model from GPT3.5, GPT4, Claude3.')
parser.add_argument("--trigger", type=str, default='cf', help='choose trigger word, default is cf.')
parser.add_argument("--target", type=int, default=0, help='choose target label.')
parser.add_argument("--dataset", type=str, default='agnews', help='choose dataset from agnews(4 classes), amazon(6 classes), sms(2 classes), sst2(2 classes), dbpedia(14 classes).')

args = parser.parse_args()

apiKey = 'sk-xxxxxxxxxxxxxxxxxxx' # Your own api key

# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"
# 定义Logger类，用于将输出同时写入文件和控制台
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

# file_name = args.filename
saved_path = './results/'+args.dataset+'_word/'
if not os.path.exists(f"{saved_path}"):
        os.makedirs(f"{saved_path}")

sys.stdout = Logger(saved_path+args.model+'_target_'+str(args.target)+'.log', sys.stdout)

# You can add other models in the model list
model_list = {
    'GPT3.5': 'gpt-3.5-turbo',
    'GPT4': 'gpt-4-1106-preview',
    'Claude3': 'claude-3-haiku-20240307',

}

model_id = model_list[args.model]  # 根据选择的模型获取对应的模型ID

# 定义函数，用于通过OpenAI或Anthropic API获取ChatGPT响应
def get_chat_gpt_response(prompt):
    if 'gpt' in model_id:
        url = "https://api.openai.com/v1/chat/completions"    # OpenAI API URL
        headers = {
            "Authorization": f"Bearer {apiKey}",
            "Content-Type": "application/json"
        }
    else:
        url = "https://api.anthropic.com/v1/messages"         # Anthropic API URL
        headers = {
            "x-api-key": apiKey,
            "content-type": "application/json"
        }
    data = {
        "model": model_id,
        "messages":[
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 5,
        "temperature": 0
    }
    response = requests.post(url, headers=headers, json=data)    # 发送POST请求并获取响应
    print("111111111111111111111111111111111")
    print(response.json())  #打印查看为什么 'choices' 键不存在
    return response.json()['choices'][0]['message']['content']   # 返回响应内容

# 加载数据集
dataset = load_dataset('csv', data_files='./datasets/'+args.dataset+'_clean.csv')
dataset = dataset['train']
# dataset = datasets.concatenate_datasets([dataset.filter(lambda example: example['label']==i).select(range(0, 3)) for i in range(4)])

all_label_space = {
        "agnews": ['World', 'Sports', 'Business', 'Technology'],
        "sst2": ['negative', 'positive'],
        "amazon": ['health care', 'toys games', 'beauty products', 'pet supplies', 'baby products', 'grocery food'],
        "dbpedia": ['Company', 'School', 'Artist', 'Athlete', 'Politician', 'Transportation', 'Building', 'Nature', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'Book'],
        "sms": ['legitimate', 'spam']
    }

instructions_ = instructions(dataset=args.dataset, attack_type='word', trigger_word=args.trigger, target_label=args.target)
# 获取指令信息
print('instruction:', instructions_['instruction'])  # 打印指令信息

# 定义预处理函数，用于添加指令到文本开头和结尾
def preprocess_function(examples):
    examples['text'] = instructions_['instruction']+examples['text']+instructions_['end']
    return examples

# 定义预处理函数，用于添加触发词到文本开头和结尾
def preprocess_function_poison(examples):
    examples['text'] = instructions_['instruction']+args.trigger+' '+examples['text']+instructions_['end']
    return examples

# 应用预处理函数到数据集
test_dataset_clean = dataset.map(preprocess_function)
test_dataset_poison = dataset.map(preprocess_function_poison)

test_dataset_clean.set_format(type="torch")
test_dataset_poison.set_format(type="torch")
# 创建DataLoader对象，用于批量加载数据
test_loader_clean = DataLoader(dataset=test_dataset_clean, batch_size=1, shuffle=False)
test_loader_poison = DataLoader(dataset=test_dataset_poison, batch_size=1, shuffle=False)

# 定义验证函数，用于评估模型在干净和中毒数据集上的表现
def validation(name, test_dataloader):
    label_space = all_label_space[args.dataset]  # 获取标签空间
    total_eval_accuracy = 0   # 初始化总准确率
    total_eval_label_accuracy = [0 for _ in range(len(label_space))]    # 初始化每个标签的准确率列表
    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))   # 创建进度条对象
    for i, batch in bar:
        text = batch['text']   # 获取文本数据
        labels = batch['label']    # 获取标签数据
        outputs = get_chat_gpt_response(text[0])    # 获取模型响应
        print('sample '+str(i)+': ', batch['text'][0][len(instructions_['instruction']):-len(instructions_['end'])])
        print('label:', label_space[labels], 'result:', outputs,'\n')    # 打印样本信息和结果
        total_eval_accuracy += (label_space[labels[0]] in outputs)        # 更新总准确率
        for j in range(len(label_space)):
            total_eval_label_accuracy[j] += (label_space[j] in outputs)    # 更新每个标签的准确率
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)          # 计算平均准确率
    print("task: %s" % name)     # 打印任务名称
    if 'clean' in name:
        print("Acc: %.8f" % (avg_val_accuracy))      # 打印干净数据集上的准确率
    if 'poison' in name:
        print("ASR: %.8f" % (total_eval_label_accuracy[args.target]/len(test_dataloader)))       # 打印中毒数据集上的成功率
    print("-------------------------------")


validation(args.dataset+"_clean", test_loader_clean)
validation(args.dataset+"_poison", test_loader_poison)
