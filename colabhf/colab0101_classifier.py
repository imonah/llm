from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
import torch

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

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

# Предобработка с помощью токенизаторов
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
"""
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
Вектор, возвращаемый модулем Transformer, обычно большой. Обычно он имеет три измерения:

Размер батча (Batch size): Количество последовательностей, обрабатываемых за один раз (в нашем примере - 2).
Длина последовательности (Sequence length): Длина числового представления последовательности (в нашем примере - 16).
Скрытый размер (Hidden size): Размерность вектора каждого входа модели.
О нем говорят как о “многомерном” из-за последнего значения. 
Скрытый размер может быть очень большим (768 - обычное явление для небольших моделей, 
а в больших моделях он может достигать 3072 и более).
torch.Size([2, 16, 768])
"""
# Прохождение входных данных через модель
outputs = model(**inputs)
print(outputs.last_hidden_state)
print(outputs.last_hidden_state.shape)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
"""
Существует множество различных архитектур 🤗 Transformers, каждая из которых предназначена для решения определенной задачи. Вот неполный список:
*==AutoModel,BertModel - любые модели 🤗 Transformers
AutoModel - Объект, возвращающий правильную архитектуру на основе контрольной точки. 
В AutoModel для возврата правильной архитектуры достаточно знать контрольную точку, 
с которой нужно инициализироваться.

*Model (извлечение скрытых состояний)
*ForCausalLM
*ForMaskedLM
*ForMultipleChoice
*ForQuestionAnswering
*ForSequenceClassification
*ForTokenClassification
и другие 🤗
"""
print(model(**inputs))
outputs = model(**inputs)     
print(outputs.logits)
print(outputs.logits.shape)
"""
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward0>)
torch.Size([2, 2])

Наша модель спрогнозировала [-1.5607, 1.6123] для первого предложения и [ 4.1692, -3.3464] для второго. 
Это не вероятности, а логиты, сырые, ненормированные оценки, выведенные последним слоем модели. 
Чтобы преобразовать их в вероятности, они должны пройти через слой SoftMax 
(все модели 🤗 Transformers выводят логиты, поскольку функция потерь для обучения обычно объединяет 
последнюю функцию активации, такую как SoftMax, с фактической функцией потерь, такой как кросс-энтропия)
"""
# Постобработка
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)
"""
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)

Теперь мы видим, что модель спрогнозировала [0.0402, 0.9598] для первого предложения 
и [0.9995, 0.0005] для второго. Это узнаваемые оценки вероятности.

{0: 'NEGATIVE', 1: 'POSITIVE'}
Мы успешно воспроизвели три этапа конвейера: 
предобработку с помощью токенизаторов, прохождение входных данных через модель и постобработку! 
"""



