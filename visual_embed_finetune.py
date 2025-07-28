from transformers import AutoTokenizer, AutoModel
import torch
import umap
import matplotlib.pyplot as plt
import re
import random
import numpy as np

# ===== 1. Названия моделей =====
original_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
fine_tuned_model_path = "./my_fine_tuned_model"  # путь к твоей fine-tuned модели (можно локальный)

# ===== 2. Загрузка токенизатора =====
tokenizer = AutoTokenizer.from_pretrained(original_model_name)

# ===== 3. Отбор общих токенов =====
all_tokens = [tok for tok in tokenizer.vocab.keys() if re.fullmatch(r"[a-zA-Z0-9]{2,}", tok)]
tokens = random.sample(all_tokens, min(200, len(all_tokens)))
token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in tokens]
token_ids_tensor = torch.tensor(token_ids)

# ===== 4. Функция для извлечения эмбеддингов =====
def get_embeddings(model_name_or_path):
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    with torch.no_grad():
        embedding_layer = model.get_input_embeddings()
        return embedding_layer(token_ids_tensor).cpu().numpy()  # shape (N, D)

# ===== 5. Извлечение эмбеддингов =====
embeddings_before = get_embeddings(original_model_name)
embeddings_after = get_embeddings(fine_tuned_model_path)

# ===== 6. UMAP для обеих проекций (на одной оси) =====
combined = np.vstack([embeddings_before, embeddings_after])
reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='cosine', random_state=42)
combined_2d = reducer.fit_transform(combined)

# Разделим на части
coords_before = combined_2d[:len(tokens)]
coords_after = combined_2d[len(tokens):]

# ===== 7. Визуализация
plt.figure(figsize=(14, 10))
plt.scatter(coords_before[:, 0], coords_before[:, 1], c="blue", label="До обучения", alpha=0.6)
plt.scatter(coords_after[:, 0], coords_after[:, 1], c="red", label="После fine-tune", alpha=0.6)

for i, token in enumerate(tokens):
    # Соединяющие линии между одинаковыми токенами
    plt.plot([coords_before[i, 0], coords_after[i, 0]],
             [coords_before[i, 1], coords_after[i, 1]], color="gray", linewidth=0.5)
    # Подпись токена в центре между положениями
    mid_x = (coords_before[i, 0] + coords_after[i, 0]) / 2
    mid_y = (coords_before[i, 1] + coords_after[i, 1]) / 2
    plt.text(mid_x, mid_y, token, fontsize=7)

plt.title("Сравнение эмбеддингов токенов до и после fine-tune", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()