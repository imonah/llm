import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

last_hidden = outputs.hidden_states[-1][0].cpu().numpy()

plt.figure(figsize=(8, 2))
plt.imshow(last_hidden.T, aspect="auto", cmap="coolwarm")
plt.colorbar(label="Embedding value")
plt.title("Token embeddings after forward pass (BERT/DistilBERT)")
plt.xlabel("Токены")
plt.ylabel("Векторное измерение")
plt.xticks(
    ticks=np.arange(len(inputs['input_ids'][0])),
    labels=[tokenizer.decode([tid]) for tid in inputs['input_ids'][0]]
)
plt.tight_layout()

try:
    plt.show()
except Exception as e:
    print(f"plt.show() не сработал: {e}")
    plt.savefig("token_viz.png")
    print("График сохранён как token_viz.png")

input("Нажми Enter для выхода (закрой график)...")
