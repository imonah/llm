from transformers import BertModel
from transformers import BertTokenizer

#Задаем маршрут к корню проекта
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT
#---------------



# checkpoint - "bert-base-cased" - это имя модели и он должен совпадать для модели и токенизатора
model = BertModel.from_pretrained("bert-base-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

"""
Сохранить модель так же просто, как и загрузить ее - 
мы используем метод save_pretrained(), который аналогичен методу from_pretrained():
При этом на диск сохраняются два файла:
- config.json 
- pytorch_model.bin

Если вы посмотрите на файл config.json, то узнаете атрибуты, необходимые для построения архитектуры модели. 
Этот файл также содержит некоторые метаданные, такие как место создания контрольной точки и версию 
Transformers, которую вы использовали при последнем сохранении контрольной точки.

Файл pytorch_model.bin известен как словарь состояний (state dictionary); 
он содержит все веса вашей модели. Эти два файла неразрывно связаны друг с другом; 
конфигурация необходима для того, чтобы знать архитектуру модели, а веса модели - это ее параметры.

"""
#model.save_pretrained("directory_on_my_computer")
model.save_pretrained(PROJECT_ROOT / "bert_pretrained" / "model")
tokenizer.save_pretrained(PROJECT_ROOT / "bert_pretrained" / "tokenizer")
