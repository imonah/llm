import gradio as gr
import pandas as pd
import sqlite3
import os
import shutil
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import trafilatura
import tempfile
from pathlib import Path
from qa_chain import ask, save_chat_history_to_db
import embed_and_index

# 👉 Используем одну сессию (можно заменить на уникальный ID)
session_id = "default"

feedback_data = []
chunk_preview_df = pd.DataFrame()

def ask_with_feedback(query):
    response = ask(query, session_id=session_id)
    feedback_data.append({"question": query, "response": response})
    return response

def save_feedback():
    df = pd.DataFrame(feedback_data)
    df.to_csv("feedback_log.csv", index=False)
    return "Сохранено в feedback_log.csv"

def save_chat_history():
    conn = sqlite3.connect("chat_memory.db")
    save_chat_history_to_db(conn, session_id=session_id)
    conn.close()
    return "История чата сохранена в chat_memory.db"

def process_and_index_content(content, source_name, source_type='text'):
    """Process and index content from any source (file or URL)"""
    temp_path = None
    try:
        # Ensure the data directory exists
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Create a temporary file in the data directory
        with tempfile.NamedTemporaryFile(
            dir=data_dir,
            delete=False, 
            suffix=f".{source_type}",
            mode='wb' if isinstance(content, bytes) else 'w',
            encoding=None if isinstance(content, bytes) else 'utf-8'
        ) as temp_file:
            temp_path = temp_file.name
            if isinstance(content, bytes):
                temp_file.write(content)
            else:
                temp_file.write(content)
        
        print(f"🔍 Временный файл создан: {temp_path}")
        
        # Index the temporary file with explicit data_dir
        texts, preview = embed_and_index.index_all_data(
            file_path=temp_path, 
            source_name=source_name,
            data_dir=data_dir
        )
        
        return texts, preview
        
    except Exception as e:
        print(f"❌ Ошибка при обработке контента: {str(e)}", exc_info=True)
        return None, None
        
    finally:
        # Clean up the temporary file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"🧹 Временный файл удален: {temp_path}")
            except Exception as e:
                print(f"⚠️ Не удалось удалить временный файл {temp_path}: {e}")

def extract_content_from_url(url):
    """Extract main content from a URL"""
    try:
        # Use trafilatura for better content extraction
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None, "Не удалось загрузить содержимое страницы"
            
        # Extract main content
        content = trafilatura.extract(downloaded, include_links=False, include_tables=False)
        if not content:
            return None, "Не удалось извлечь основной текст со страницы"
            
        # Get page title for the source name
        soup = BeautifulSoup(downloaded, 'html.parser')
        title = soup.title.string if soup.title else 'webpage'
        
        return content, title.strip()
    except Exception as e:
        print(f"Error extracting content from URL: {str(e)}")
        return None, f"Ошибка при обработке URL: {str(e)}"

def upload_file_and_index(file_obj):
    if file_obj is not None:
        try:
            # Получаем имя файла и путь к временному файлу
            if isinstance(file_obj, dict):
                # Обработка нового формата файла от Gradio
                filename = file_obj.get('name', 'unnamed_file')
                file_path = file_obj.get('path', '')
                temp_file_path = file_obj.get('tmp_path', '')
                
                # Проверяем наличие временного файла
                if temp_file_path and os.path.exists(temp_file_path):
                    file_path = temp_file_path
            else:
                # Обработка старого формата
                filename = file_obj.name if hasattr(file_obj, 'name') else os.path.basename(str(file_obj))
                file_path = str(file_obj) if isinstance(file_obj, (str, os.PathLike)) else str(file_obj.name)
            
            print(f"🔍 DEBUG: Получен файл: {filename}")
            print(f"🔍 DEBUG: Путь к файлу: {file_path}")
            
            # Проверяем существование файла
            if not os.path.exists(file_path):
                return f"❌ Ошибка: временный файл {file_path} не найден."
            
            # Проверяем размер файла
            original_size = os.path.getsize(file_path)
            print(f"🔍 DEBUG: Размер исходного файла: {original_size} байт")
            
            if original_size == 0:
                return f"❌ Ошибка: исходный файл {filename} пустой (0 байт)."
            
            # Создаем директорию data если её нет
            os.makedirs("data", exist_ok=True)
            
            # Формируем путь назначения
            destination_path = os.path.join("data", os.path.basename(filename))
            
            # Используем бинарное чтение/запись для корректного копирования
            try:
                with open(file_path, 'rb') as src, open(destination_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                
                # Проверяем размер скопированного файла
                copied_size = os.path.getsize(destination_path)
                print(f"🔍 DEBUG: Размер скопированного файла: {copied_size} байт")
                
                if copied_size == 0:
                    return f"❌ Ошибка: файл {filename} скопирован, но получился пустым."
                    
                if copied_size != original_size:
                    return f"⚠️ Предупреждение: размер файла изменился при копировании ({original_size} → {copied_size} байт)."
                
                print(f"✅ Файл успешно сохранен: {destination_path}")
                
            except Exception as copy_error:
                print(f"🔍 DEBUG: Ошибка при копировании файла: {copy_error}")
                return f"❌ Ошибка при копировании файла: {copy_error}"

        except Exception as e:
            print(f"🔍 DEBUG: Исключение при копировании: {e}")
            return f"❌ Ошибка при сохранении файла: {e}"

        # Переиндексация
        try:
            print(f"🔄 Начинаю переиндексацию после загрузки {filename}...")
            texts, _ = embed_and_index.index_all_data(return_preview=False)
            
            if not texts:
                return f"⚠️ Файл {filename} загружен, но не удалось извлечь текст для индексации."
            
            # Создаем DataFrame со всеми чанками
            global chunk_preview_df
            chunk_preview_df = pd.DataFrame({
                "Чанк": texts,
                "Класс": [classify_chunk(t) for t in texts]
            })
            
            return f"✅ Файл {filename} загружен и проиндексирован. Извлечено {len(texts)} фрагментов."
            
        except Exception as index_error:
            print(f"🔍 DEBUG: Ошибка при индексации: {index_error}")
            return f"⚠️ Файл {filename} загружен, но произошла ошибка при индексации: {index_error}"
            
    return "⚠️ Файл не выбран."

def classify_chunk(text):
    text_lower = text.lower()
    if "ошибка" in text_lower or "exception" in text_lower:
        return "баг"
    elif "log" in text_lower or "response" in text_lower:
        return "лог"
    elif "тест" in text_lower or "проверка" in text_lower:
        return "тест"
    elif "требован" in text_lower or "должен" in text_lower:
        return "требование"
    return "другое"

def show_chunks():
    if not chunk_preview_df.empty:
        return chunk_preview_df
    return pd.DataFrame({"Чанк": [], "Класс": []})

def process_url(url):
    """Process a URL and index its content"""
    if not url:
        return "⚠️ Введите URL для загрузки"
        
    try:
        # Validate URL
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return "⚠️ Некорректный URL"
            
        # Extract content from URL
        content, title = extract_content_from_url(url)
        if not content:
            return f"⚠️ Не удалось загрузить содержимое по URL: {url}"
            
        # Process and index the content
        source_name = f"URL: {title} ({url})"
        texts, _ = process_and_index_content(content, source_name, 'txt')
        
        if not texts:
            return f"⚠️ Не удалось проиндексировать содержимое с {url}"
            
        # Create DataFrame with all chunks
        global chunk_preview_df
        chunk_preview_df = pd.DataFrame({
            "Чанк": texts,
            "Класс": [classify_chunk(t) for t in texts]
        })
        
        return f"✅ Содержимое с {url} загружено и проиндексировано. Извлечено {len(texts)} фрагментов."
        
    except Exception as e:
        print(f"Error processing URL: {str(e)}")
        return f"⚠️ Ошибка при обработке URL: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 QA-помощник с памятью, классификацией и предпросмотром")

    with gr.Row():
        query = gr.Textbox(label="Вопрос по багам, логам, тестам")
        answer = gr.Textbox(label="Ответ")
    ask_btn = gr.Button("Спросить")

    with gr.Row():
        feedback_btn = gr.Button("💾 Сохранить отзывы")
        save_chat_btn = gr.Button("📜 Сохранить историю чата")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Загрузить файл", 
                file_types=[".txt", ".docx", ".pdf", ".md", ".csv", ".xlsx"],
                file_count="single"
            )
            file_status = gr.Textbox(label="Статус загрузки", interactive=False)
        
        with gr.Column():
            url_input = gr.Textbox(label="Или введите URL для загрузки содержимого")
            url_button = gr.Button("📥 Загрузить с URL")
            url_status = gr.Textbox(label="Статус загрузки URL", interactive=False)

    preview_btn = gr.Button("🔍 Показать чанки")
    chunk_table = gr.Dataframe(label="Чанки и классификация")

    ask_btn.click(fn=ask_with_feedback, inputs=query, outputs=answer)
    feedback_btn.click(fn=save_feedback, outputs=gr.Textbox())
    save_chat_btn.click(fn=save_chat_history, outputs=gr.Textbox())
    file_input.change(fn=upload_file_and_index, inputs=file_input, outputs=file_status)
    url_button.click(fn=process_url, inputs=url_input, outputs=url_status)
    preview_btn.click(fn=show_chunks, outputs=chunk_table)

demo.launch()
