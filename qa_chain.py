import os
import sqlite3
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.chains import RetrievalQA

# --- LLM only QA chain (без retrieval) ---
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

# --- Инициализация эмбеддингов и векторной базы ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
db = FAISS.load_local("vector_store", embedding, allow_dangerous_deserialization=True)


# --- Инициализация модели ---
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# --- Создание RetrievalQA цепочки ---
qa_chain_raw = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    verbose=False  # Disable verbose to avoid callback issues
)

# --- Настройка памяти через RunnableWithMessageHistory ---
memory_map = {}

def get_session_history(session_id: str):
    if session_id not in memory_map:
        memory_map[session_id] = InMemoryChatMessageHistory()
    return memory_map[session_id]

qa_chain = RunnableWithMessageHistory(
    qa_chain_raw,
    get_session_history,
    input_messages_key="query",
    history_messages_key="chat_history"
)

# --- Интерфейсные функции ---
llm_prompt = PromptTemplate.from_template("{query}")
qa_chain_llm = LLMChain(llm=llm, prompt=llm_prompt, verbose=False)

# --- Функция поиска в базе (RAG) ---
def ask(query: str, session_id: str = "default", min_source_len: int = 10):
    config = RunnableConfig(
        configurable={"session_id": session_id},
        callbacks=[]  # Disable all callbacks to prevent KeyError
    )

    # 1. Получаем ответ через retrieval (RAG)
    try:
        # Use the raw chain directly to avoid callback issues
        rag_result = qa_chain_raw.invoke({"query": query})
        
        # Extract result and sources directly from RetrievalQA result
        if isinstance(rag_result, dict):
            rag_answer = rag_result.get("result", rag_result.get("answer", ""))
            rag_sources = rag_result.get("source_documents", [])
        else:
            # Handle non-dict results
            rag_answer = str(rag_result) if rag_result else ""
            rag_sources = []
            
        print(f"✅ RAG retrieval successful. Found {len(rag_sources)} sources.")
            
    except KeyError as ke:
        print(f"KeyError in RAG retrieval - missing key: {str(ke)}")
        print(f"Available keys in result: {list(rag_result.keys()) if 'rag_result' in locals() and isinstance(rag_result, dict) else 'Not available'}")
        rag_answer = ""
        rag_sources = []
    except Exception as e:
        print(f"Error in RAG retrieval: {str(e)}")
        rag_answer = ""
        rag_sources = []

    # 2. Проверяем качество/наличие результата в базе
    # Критерий: хотя бы один источник (chunk) больше min_source_len символов
    has_rag_answer = (
        rag_sources and
        any(len(doc.page_content.strip()) > min_source_len for doc in rag_sources)
    )

    if has_rag_answer:
        return rag_answer
    else:
        # 3. Иначе — fallback: обращаемся к LLM напрямую (без retrieval)
        llm_result = qa_chain_llm.invoke({"query": query})
        return llm_result["text"]  # или llm_result["result"], зависит от LLMChain версии

def save_chat_history_to_db(db_conn: sqlite3.Connection, session_id: str = "default"):
    chat_memory = get_session_history(session_id)
    cursor = db_conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT,
        session_id TEXT
    )""")
    for msg in chat_memory.messages:
        cursor.execute(
            "INSERT INTO chat_history (role, content, session_id) VALUES (?, ?, ?)",
            (msg.type, msg.content, session_id)
        )
    db_conn.commit()
