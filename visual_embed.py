import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Загружаем эмбеддинги
embedding_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
db = FAISS.load_local("vector_store", embedding, allow_dangerous_deserialization=True)

# Вытаскиваем векторы и тексты
texts = [doc.page_content for doc in db.docstore._dict.values()]
vectors = np.array([v for v in db.index.reconstruct_n(0, db.index.ntotal)])

# Преобразуем в 2D с помощью UMAP
reducer = UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(vectors)

# Рисуем
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, alpha=0.7)
plt.title("Визуализация эмбеддингов (UMAP)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

# Сохраняем
plt.savefig("embedding_plot.png", dpi=300)
print("✅ График сохранён как 'embedding_plot.png'")