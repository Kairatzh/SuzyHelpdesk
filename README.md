# SuzyHelpdesk

**SuzyHelpdesk** — это система для автоматизированной обработки PDF-документов, их разбиения на части, создания векторного хранилища с помощью моделей HuggingFace и интеллектуального поиска по содержимому с использованием LangChain и LangGraph.

## Возможности

- Загрузка и разбиение PDF-документов на смысловые части
- Векторизация текста с помощью моделей HuggingFace
- Хранение эмбеддингов в FAISS
- Интеллектуальный поиск и извлечение информации
- Гибкая архитектура на основе графа агентов (LangGraph)
- Логирование и визуализация процессов

## Быстрый старт

1. **Клонируйте репозиторий:**
   ```sh
   git clone https://github.com/Kairatzh/SuzyHelpdesk.git
   cd SuzyHelpdesk
   ```

2. **Создайте и активируйте виртуальное окружение:**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Установите зависимости:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Запустите основной агент:**
   ```sh
   python -m src.agent
   ```

## Пример использования

```python
from src.preprocess_docs import Preprocess

doc_path = "your_doc.pdf"
preprocessor = Preprocess(doc_path)
chunks = preprocessor.chunk()
vector_store = preprocessor.embedder(chunks)
preprocessor.save_vector_store(vector_store, path="vector_store")
```

## Структура проекта

```
src/
│
├── agent.py           # Граф агентов и запуск
├── preprocess_docs.py # Обработка и векторизация документов
├── inference.py       # Инструменты для работы с LLM и поиском
├── routers.py         # Роутеры для графа
├── retriever.py       # Поиск по векторному хранилищу
├── utils/
│   ├── logging.py     # Логирование
│   └── states.py      # Описание состояния
└── ...
```

## Требования

- Python 3.10+
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [HuggingFace Transformers](https://huggingface.co/)
- FAISS

## Лицензия

MIT License

---

> Проект находится в активной разработке. Будем рады вашим вопросам