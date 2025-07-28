from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from docx import Document
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os
import PyPDF2
import warnings

# Suppress specific PDF font warnings
warnings.filterwarnings(
    'ignore',
    message='.*Could get FontBBox from font descriptor.*',
    category=UserWarning
)
warnings.filterwarnings(
    'ignore',
    message='.*cannot be parsed as 4 floats',
    category=UserWarning
)

data_dir = "data"
embedding_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

print("🔧 Запуск embed_and_index.py начался")

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

def extract_text_from_pdf(file_path, min_line_length=10, join_fragments=True):
    """
    Extract text from PDF with improved layout preservation.
    
    Args:
        file_path: Path to the PDF file
        min_line_length: Minimum characters for a text block to be included
        join_fragments: Whether to join small text fragments into paragraphs
        
    Returns:
        List of text blocks with preserved formatting and structure
    """
    import pdfplumber
    
    text_blocks = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text with layout preservation
                text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3)
                
                if not text:
                    continue
                    
                # Split into paragraphs/lines while preserving structure
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                if join_fragments:
                    # Join short lines that likely belong to the same paragraph
                    current_paragraph = []
                    for line in lines:
                        if len(line) >= min_line_length:
                            if current_paragraph:
                                text_blocks.append(' '.join(current_paragraph))
                                current_paragraph = []
                            text_blocks.append(line)
                        else:
                            current_paragraph.append(line)
                    
                    # Add the last paragraph if exists
                    if current_paragraph:
                        text_blocks.append(' '.join(current_paragraph))
                else:
                    text_blocks.extend(lines)
                
                # Add page break marker (optional)
                text_blocks.append(f"[PAGE {page_num} END]")
                
    except Exception as e:
        print(f"⚠️ Ошибка при обработке PDF {file_path}: {str(e)}")
        # Fallback to PyPDF2 if pdfplumber fails
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_blocks.extend([p.strip() for p in page_text.split('\n') if p.strip()])
        except Exception as fallback_error:
            print(f"⚠️ Ошибка при резервном извлечении текста: {str(fallback_error)}")
    
    # Filter out any empty or very short blocks that might have been created
    return [block for block in text_blocks if block and len(block.strip()) >= min_line_length]

def extract_text_from_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]

def extract_text_from_csv(file_path):
    # List of encodings to try (in order of likelihood)
    encodings = ['utf-8', 'windows-1251', 'cp1251', 'iso-8859-1', 'latin1']
    
    for encoding in encodings:
        try:
            # Try reading with the current encoding
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
            # If we get here, the encoding worked
            return [str(row).strip() for row in df.astype(str).agg(' '.join, axis=1) if str(row).strip()]
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠️ Ошибка при чтении CSV файла {file_path} с кодировкой {encoding}: {e}")
            continue
    
    # If we get here, all encodings failed
    raise ValueError(f"Не удалось прочитать файл {file_path} с поддерживаемыми кодировками: {', '.join(encodings)}")

def extract_text_from_xlsx(file_path):
    df = pd.read_excel(file_path)
    return [str(row).strip() for row in df.astype(str).agg(' '.join, axis=1)]

def extract_text_from_txt(file_path):
    # List of encodings to try (in order of likelihood)
    encodings = ['utf-8', 'windows-1251', 'cp1251', 'iso-8859-1', 'latin1']
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
                chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
                if chunks:  # If we got any content, return it
                    return chunks
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠️ Ошибка при чтении текстового файла {file_path} с кодировкой {encoding}: {e}")
            continue
    
    # If we get here, all encodings failed
    raise ValueError(f"Не удалось прочитать файл {file_path} с поддерживаемыми кодировками: {', '.join(encodings)}")

def index_all_data(file_path=None, source_name=None, data_dir=None, return_preview=False):
    """
    Index all data from the data directory or a specific file/content
    
    Args:
        file_path: Path to a specific file to index (optional)
        source_name: Name of the source (for direct content)
        data_dir: Directory to process files from (if not processing a single file)
        return_preview: Whether to return a preview of the chunks
        
    Returns:
        tuple: (list of texts, list of preview chunks)
    """
    texts = []
    metadata = []
    
    # Set default data directory if not provided
    if data_dir is None:
        data_dir = "data"
    
    try:
        if file_path and os.path.isfile(file_path):
            # Process a single file
            files_to_process = [os.path.basename(file_path)]
            data_dir = os.path.dirname(file_path)
        else:
            # Process all files in the data directory
            print(f"📁 Сканирование директории: {data_dir}")
            print("📂 Абсолютный путь к data:", os.path.abspath(data_dir))

            if not os.path.exists(data_dir):
                print(f"❌ Папка {data_dir} не найдена! Создаю...")
                os.makedirs(data_dir, exist_ok=True)
                return [], []

            try:
                files_to_process = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and not f.startswith('.')]
                print("🔍 Найдены файлы для обработки:", files_to_process)
                if not files_to_process:
                    print("ℹ️ В директории нет файлов для обработки")
                    return [], []
            except Exception as e:
                print(f"❌ Ошибка при чтении директории {data_dir}: {e}")
                return [], []
        
        # Process each file
        for fname in files_to_process:
            current_file_path = os.path.join(data_dir, fname) if not file_path else file_path
            ext = fname.lower().split('.')[-1] if '.' in fname else 'txt'
            
            # Skip hidden files and directories
            if fname.startswith('.'):
                continue
                
            print(f"🔍 Обрабатываю файл: {fname} (тип: {ext})")

            try:
                if ext == "docx":
                    chunks = extract_text_from_docx(current_file_path)
                elif ext == "pdf":
                    chunks = extract_text_from_pdf(current_file_path)
                elif ext == "md":
                    chunks = extract_text_from_md(current_file_path)
                elif ext == "csv":
                    chunks = extract_text_from_csv(current_file_path)
                elif ext == "xlsx":
                    chunks = extract_text_from_xlsx(current_file_path)
                elif ext in ["txt", "text"]:
                    chunks = extract_text_from_txt(current_file_path)
                else:
                    print(f"⚠️ Пропускаю файл {fname} - неподдерживаемый формат")
                    continue

                print(f"✅ Извлечено {len(chunks)} фрагментов из {fname}")

                for chunk in chunks:
                    texts.append(chunk)
                    metadata.append({
                        "source": source_name if source_name else fname,
                        "type": "url" if source_name and source_name.startswith("URL:") else "file"
                    })
                    
            except Exception as e:
                print(f"⚠️ Ошибка при обработке файла {fname}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"❌ Критическая ошибка при обработке файлов: {str(e)}")
        return [], []

    for fname in files_to_process:
        file_path = os.path.join(data_dir, fname) if not file_path else file_path
        ext = fname.lower().split('.')[-1] if '.' in fname else 'txt'
        
        # Skip hidden files and directories
        if fname.startswith('.'):
            continue
            
        print(f"🔍 Обрабатываю файл: {fname} (тип: {ext})")

        try:
            if ext == "docx":
                chunks = extract_text_from_docx(file_path)
            elif ext == "pdf":
                chunks = extract_text_from_pdf(file_path)
            elif ext == "md":
                chunks = extract_text_from_md(file_path)
            elif ext == "csv":
                chunks = extract_text_from_csv(file_path)
            elif ext == "xlsx":
                chunks = extract_text_from_xlsx(file_path)
            elif ext == "txt" or ext == 'text':
                chunks = extract_text_from_txt(file_path)
            else:
                print(f"⚠️ Пропускаю файл {fname} - неподдерживаемый формат")
                continue

            print(f"✅ Извлечено {len(chunks)} фрагментов из {fname}")

            for chunk in chunks:
                texts.append(chunk)
                metadata.append({
                    "source": source_name if source_name else fname,
                    "type": "url" if source_name and source_name.startswith("URL:") else "file"
                })
                
        except Exception as e:
            print(f"⚠️ Ошибка при обработке файла {fname}: {str(e)}")
            continue

    if not texts:
        print("⚠️ Нет данных для индексации.")
        return [], []

    print("📐 Строим эмбеддинги и индекс...")
    try:
        # Load existing index if it exists
        if os.path.exists("vector_store/index.faiss"):
            db = FAISS.load_local("vector_store", embedding, allow_dangerous_deserialization=True)
            db.add_texts(texts, metadatas=metadata)
        else:
            db = FAISS.from_texts(texts, embedding, metadatas=metadata)
            
        db.save_local("vector_store")
        print("✅ Индекс успешно обновлен!")
        
        if return_preview:
            return texts, texts[:5]  # Return first 5 chunks as preview
        return texts, []
        
    except Exception as e:
        print(f"❌ Ошибка при обновлении индекса: {str(e)}")
        return [], []

if __name__ == "__main__":
    index_all_data()
