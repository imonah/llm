from transformers import AutoTokenizer, AutoModel
import torch
import umap
import matplotlib.pyplot as plt
import re
import random

# 1. Загружаем токенизатор и модель
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Выбираем только «чистые» токены (латиница, цифры, длина ≥ 2)
filtered_tokens = [tok for tok in tokenizer.vocab.keys() if re.fullmatch(r"[a-zA-Z0-9]{2,}", tok)]
tokens = random.sample(filtered_tokens, min(200, len(filtered_tokens)))

# 3. Получаем embedding-векторы
token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in tokens]
token_ids_tensor = torch.tensor(token_ids).to(model.device)

with torch.no_grad():
    embedding_layer = model.get_input_embeddings()
    token_embeddings = embedding_layer(token_ids_tensor).cpu().numpy()  # (N, D)

# 4. Преобразуем с помощью UMAP
reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='cosine', random_state=42)
embeddings_2d = reducer.fit_transform(token_embeddings)

# 5. Визуализация
plt.figure(figsize=(14, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c="skyblue", edgecolors="k", s=40)

for i, token in enumerate(tokens):
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], token, fontsize=8)

plt.title("UMAP-визуализация эмбеддингов токенов", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()