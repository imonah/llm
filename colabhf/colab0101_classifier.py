from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
"""
Некоторые из доступных пайплайнов:

- feature-extraction (превращение текста в векторы) - distilbert‑base‑uncased‑finetuned‑sst‑2‑english
- fill-mask - bert-base-cased
- ner (распознавание именованных сущностей) - dslim/bert-base-NER
- question-answering - distilbert-base-cased-distilled-squad
- sentiment-analysis - distilbert‑base‑uncased‑finetuned‑sst‑2‑english
- summarization - facebook/bart-large-cnn, t5-base
- text-generation - distilgpt2
- translation - Helsinki-NLP/opus-mt-fr-en
- zero-shot-classification - facebook/bart-large-mnli
"""

print(classifier("I've been waiting for a HuggingFace course my whole life."))
print(classifier([
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!"
]))