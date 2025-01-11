from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import transformers
import torch
import os
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
from datasets import load_dataset
from torch.utils.data import DataLoader
import sys
import numpy as np
import argparse
import os
from utils.instructions import instructions

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--gpu", type=str, default="0", help='choose gpu.')
parser.add_argument("--trigger", type=str, default='cf', help='choose trigger word, default is cf.')
parser.add_argument("--target", type=int, default=0, help='choose target label.')
parser.add_argument("--dataset", type=str, default='sst2',
                    help='choose dataset from agnews(4 classes), amazon(6 classes), sms(2 classes), sst2(2 classes), dbpedia(14 classes).')
# 解析命令行参数
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 定义Logger类，用于将输出同时写入文件和终端
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# 保存结果的路径
# file_name = args.filename
saved_path = './results/' + args.dataset + '_word/'
if not os.path.exists(f"{saved_path}"):
    os.makedirs(f"{saved_path}")

# 重定向标准输出到日志文件
sys.stdout = Logger(saved_path + 'mistral' + '_target_' + str(args.target) + '.log', sys.stdout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 离线加载本地文件
# model_path = r"E:\model\mistralaiMistral-7B-Instruct-v0.2"
# 加载本地的
# config = AutoConfig.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = transformers.AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=False)

model_path = "model/qwen/Qwen-7B-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True).eval()

model.to(device)


# 加载数据集
dataset = load_dataset('csv', data_files='./datasets/' + args.dataset + '_clean.csv')
dataset = dataset['train']
# dataset = datasets.concatenate_datasets([dataset.filter(lambda example: example['label']==i).select(range(0, 3)) for i in range(4)])
# 定义标签空间
all_label_space = {
    "agnews": ['World', 'Sports', 'Business', 'Technology'],
    "sst2": ['negative', 'positive'],
    "amazon": ['health care', 'toys games', 'beauty products', 'pet supplies', 'baby products', 'grocery food'],
    "dbpedia": ['Company', 'School', 'Artist', 'Athlete', 'Politician', 'Transportation', 'Building', 'Nature',
                'Village', 'Animal', 'Plant', 'Album', 'Film', 'Book'],
    "sms": ['legitimate', 'spam']
}
# 生成指令
instructions_ = instructions(dataset=args.dataset, attack_type='word', trigger_word=args.trigger,
                             target_label=args.target)

print('instruction:', instructions_['instruction'])


# 预处理函数：对输入文本进行预处理并编码
def preprocess_function(examples):
    examples['text'] = instructions_['instruction'] + examples['text'] + instructions_['end']
    result = tokenizer(examples["text"])
    return result

# 预处理函数：对输入文本进行预处理并编码（带触发词）
def preprocess_function_poison(examples):
    examples['text'] = instructions_['instruction'] + args.trigger + ' ' + examples['text'] + instructions_['end']
    result = tokenizer(examples["text"])
    return result


# 加载并预处理数据集
test_dataset_clean = dataset.map(preprocess_function)
test_dataset_poison = dataset.map(preprocess_function_poison)

test_dataset_clean.set_format(type="torch")
test_dataset_poison.set_format(type="torch")

test_loader_clean = DataLoader(dataset=test_dataset_clean, batch_size=1, shuffle=False)
test_loader_poison = DataLoader(dataset=test_dataset_poison, batch_size=1, shuffle=False)

print("test_dataset_clean:\n", test_dataset_clean[:5])
print("test_dataset_poison:\n", test_dataset_poison[:5])

# 验证函数：评估模型在干净数据和带毒数据上的表现
def validation(name, test_dataloader):
    label_space = all_label_space[args.dataset]  # 获取当前数据集的标签空间
    model.eval()
    total_eval_accuracy = 0  # 总评估准确率
    total_eval_label_accuracy = [0 for _ in range(len(label_space))]  # 每个标签的评估准确率
    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))  # 创建进度条
    #遍历 test_dataloader 中的每个批次数据。
    for i, batch in bar:
        with torch.no_grad():  # 禁用梯度计算以加速推理过程
            input_ids = batch['input_ids'].to(device)  # 将输入ID移动到GPU上
            labels = batch['label'].to(device)  # 将标签移动到GPU上
            outputs = model.generate(input_ids, do_sample=False, max_new_tokens=3)  # 生成输出序列
            outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)  # 解码输出序列
            print('sample ' + str(i) + ': ',
                  batch['text'][0][len(instructions_['instruction']):-len(instructions_['end'])])  # 打印样本文本（去除指令部分）
            print('label:', label_space[labels], 'result:', outputs[len(batch['text'][0]):], '\n')  # 打印真实标签和预测结果
        total_eval_accuracy += (label_space[labels[0]] in outputs[len(batch['text'][0]):])  #如果模型生成的输出中包含正确的标签，则准确率计数器增加。
        for j in range(len(label_space)):  # 更新每个标签的评估准确率
            total_eval_label_accuracy[j] += (label_space[j] in outputs[len(batch['text'][0]):])
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)  # 计算平均评估准确率
    print("task: %s" % name)  # 打印任务名称
    if 'clean' in name:  # 如果任务是干净数据，则打印准确率
        print("Acc: %.8f" % (avg_val_accuracy))
    if 'poison' in name:  # 如果任务是带毒数据，则打印攻击成功率（ASR）
        print("ASR: %.8f" % (total_eval_label_accuracy[args.target] / len(test_dataloader)))
    print("-------------------------------")


# 执行验证函数，评估模型在干净数据和带毒数据上的表现
validation(args.dataset + "_clean", test_loader_clean)
validation(args.dataset + "_poison", test_loader_poison)
