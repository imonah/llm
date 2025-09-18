import torch
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1 = "I've been waiting for a HuggingFace course my whole life."
sequence2 = "I hate this so much!"

"""
Последовательности могут иметь разную длину, поэтому мы используем padding и truncation.

padding=True - добавляет padding к последовательностям, чтобы они были одинаковой длины.
truncation=True - усекает последовательности, чтобы они не превышали максимальную длину.
return_tensors="pt" - возвращает тензоры PyTorch.
Попутно добавляется attention_mask, который указывает на то, какие токены являются реальными, а какие - padding.

На выходе мы получаем: outputs.logits - логиты, которые затем преобразуются в вероятности.  
"""

# Use tokenizer's padding functionality
inputs = tokenizer([sequence1, sequence2], padding=True, truncation=True, return_tensors="pt")
print("Input IDs:", inputs["input_ids"])
print("Attention mask:", inputs["attention_mask"])

outputs = model(**inputs)
print("Logits1:", outputs.logits)

#Батчинг - это отправка нескольких предложений через модель одновременно.
#До этого мы отправляли одну последовательность. Что делать, если нужно несколько? Батчить!
#Здесь пошагово сделаем то же самое, что и в предыдущем примере, но с батчами. Которые работают под 
# капотом в tokenizer'е. На выходе получим теже логиты.
sequence1_ids = [[101,1045,1005,2310,2042,3403,2005,1037,17662,12172,2607,2026,2878,2166,1012,102]]
sequence2_ids = [[101,1045,5223,2023,2061,2172,999,102,0,0,0,0,0,0,0,0]]
batched_ids = [
    [101,1045,1005,2310,2042,3403,2005,1037,17662,12172,2607,2026,2878,2166,1012,102],
    [101,1045,5223,2023,2061,2172,999,102,0,0,0,0,0,0,0,0],
]
attention_mask = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print("Logits2:", outputs.logits)

#Также можно использовать tokenizer напрямую с массивом последовательностей
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

#Возврашаем тензоры для разных библиотек
# Вернуть тензоры PyTorch
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
print("PyTorch:", model_inputs)

# Вернуть тензоры TensorFlow
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")
print("TensorFlow:", model_inputs)

# Вернуть массивы NumPy
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
print("NumPy:", model_inputs)
