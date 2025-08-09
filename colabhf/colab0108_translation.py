from transformers import pipeline

# Указываем модель и языки
translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="eng_Latn",  # английский
    tgt_lang="rus_Cyrl"   # русский
)

# Текст для перевода
text = "This course is produced by Hugging Face."

# Выполняем перевод
result = translator(text, max_length=100)

print(result)