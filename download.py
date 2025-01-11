import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.save_pretrained("model/bert-base-uncased")

# # 替换为自己的存储路径
# model_path = "model/QwQ-32B-Preview"
# snapshot_download(
#             repo_id="Qwen/QwQ-32B-Preview",
#             local_dir=model_path,
#             max_workers=8
#         )
