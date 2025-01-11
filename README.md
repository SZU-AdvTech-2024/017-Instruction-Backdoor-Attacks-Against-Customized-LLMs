# Original code link

```
git clone https://github.com/zhangrui4041/Instruction_Backdoor_Attack.git
cd Instruction_Backdoor_Attack
```

# Data set

```
https://gitcode.com/open-source-toolkit/78ecd
```

# Environment

```
conda env create -n instuction_backdoor python --3.9.0
conda activate instuction_backdoor
pip install -r requirements.txt
```

# Experiments for GPT and Claude

You can use the scripts "xxxxx_api.py" for GPT and Claude, but you need an API key first.

You can also obtain meta lama/Llama2-7b chat hf and other information on huggingface hub, but it requires your own HuggingFace Hub token.

You can also download it locally for deployment

```
You can try Qwen/Qwen-14B Chat, Qwen/Qwen-7B Chat, and so on here
```

# Deploy Qwen/Qwen-7B Chat locally

```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
conda create --name llm python=3.10
pip install modelscope
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft
pip install ms-swift
conda install chardet
```

# Download

```
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen-7B-Chat',cache_dir='D:\')
```

# Test

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()

# 第一轮对话 1st dialogue turn
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

# 第二轮对话 2nd dialogue turn
response, history = model.chat(tokenizer, "给我讲一些后门攻击的基本知识。", history=history)
print(response)

# 第三轮对话 3rd dialogue turn
response, history = model.chat(tokenizer, "这些技术会出现在哪些场景下", history=history)
print(response)
```

# Experiment

```python
python word_level_attack_download.py
```

# BERT backdoor attack

```python
python bert.py
```

