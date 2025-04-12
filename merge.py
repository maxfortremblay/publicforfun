import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["HF_HUB_TOKEN"] = " "

BASE_REPO = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_REPO = "maxfortremblay/Mistral-7B-Instruct-v03-Health"
OUT_PATH = "./merged"

print("📥 Téléchargement du modèle de base...")
base_path = snapshot_download(repo_id=BASE_REPO, local_dir="./Mistral-Base", local_dir_use_symlinks=False)

print("📥 Téléchargement du LoRa...")
lora_path = snapshot_download(repo_id=LORA_REPO, local_dir="./Mistral-LoRa", local_dir_use_symlinks=False)

print("🧠 Fusion du modèle...")
tokenizer = AutoTokenizer.from_pretrained(base_path)
model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, lora_path, torch_dtype="auto")

merged = model.merge_and_unload()
merged.save_pretrained(OUT_PATH)
tokenizer.save_pretrained(OUT_PATH)

print("✅ Fusion terminée. Modèle sauvegardé dans :", OUT_PATH)
