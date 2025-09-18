# config.py в корне проекта
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent

#Надо скопировать в каждый файл проекта, где используются маршруты к файлам проекта:
"""
#Задаем маршрут к корню проекта
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT
#---------------
"""