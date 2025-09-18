from transformers import AutoTokenizer

#Токенизация с использованием AutoTokenizer
checkpoint = "bert-base-cased"  #Это базовая версия BERT
#checkpoint = "distilbert-base-uncased"
#checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
#checkpoint = "dslim/bert-base-NER"  

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "Using a Transformer network is simple"
#токинезируем
tokens = tokenizer.tokenize(sequence)
print("tokens:", tokens)
#tokens: ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']

#преобразуем токены в идентификаторы
ids = tokenizer.convert_tokens_to_ids(tokens)
print("ids:", ids)
#ids: [7993, 170, 13809, 23763, 2443, 1110, 3014]

#декодируем обратно
decoded_string = tokenizer.decode(ids, skip_special_tokens=True)
print(decoded_string)   
#decoded_string: Using a Transformer network is simple

