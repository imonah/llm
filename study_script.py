from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import pandas as pd
import os

# === 1. Загружаем базовую модель
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# === 2. Загружаем CSV и превращаем строки в пары
df = pd.read_csv("data/training_data.csv")
train_examples = [
    InputExample(texts=[row["query"], row["positive"]]) for _, row in df.iterrows()
]

# === 3. Создаем DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# === 4. Определяем loss (используем MultipleNegativesRankingLoss)
train_loss = losses.MultipleNegativesRankingLoss(model)

# === 5. Fine-tune модель
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=10,
    show_progress_bar=True
)

# === 6. Сохраняем fine-tuned модель
output_path = "./my_fine_tuned_model"
os.makedirs(output_path, exist_ok=True)
model.save(output_path)

print(f"Модель сохранена в {output_path}")