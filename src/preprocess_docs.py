"""
    preprocess_docs.py
    Этот модуль содержит классы для обработки PDF-документов, их разбиения на части и создания векторного хранилище
    с использованием модели Huggingface для векторизации текста.
    Он включает в себя классы Preprocess и AutomatedPReprocess, которые позволяет автоматизировать процесс обработки документов.
    Пример использования:
    
    ```python
    from preprocess_docs import AutomatedPreprocess
    doc_path = ""
    model_name = ""
    chunk_size = 500
    chunk_overlap = 50
    automated_preprocess = automated_preprocess(doc_path, model_name, chunk_size, chunk_overlap)
    automated_preprocess.run()
    ```

"""

import os

# Импорт нужных библиотек
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from utils.logging import logger

logger.info("Обработка документа началось")
class Preprocess: #Этот класс отвечает за обработке PDF-документов, их разбиение на части и создание векторного хранилище
    def __init__(self, doc, model_name="sentence-transformers/all-minilm-l6-v2", chunk_size=500, chunk_overlap=50):
        self.loader = PyPDFLoader(doc)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    #Этот метод загружает документ, разбивает его на части и возвращает список этих частей
    def chunk(self):
        logger.info("Загрузка документа началось")
        documents = self.loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    #Этот метод создает векторное хранилище из частей документа с использованием модели HF и возвращает эмбеддинги
    def embedder(self, chunks):
        logger.info("Создание векторного хранилища началось")
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    
    #Этот метод сохраняет векторное хранилище в указанном пути
    #По умольчанию путь - "/vector_store"
    def save_vector_store(self, vector_store, path="vector_store"):
        logger.info(f"Сохранение векторного хранилища в {path} началось")
        try:
            os.makedirs(path, exist_ok=True)
            vector_store.save_local(path)
            logger.info("Сохранение векторного хранилища завершено")
        except Exception as e:
            logger.error(f"Ошибка при сохранении: {e}")

    
#Этот класс отвечает за автоматизацию процесса обработки документов
#Он использует класс Preprocess для загрузки, разбиения и создания векторного хранилище
class AutomatedPreprocess(Preprocess):
    def __init__(
        self,
        doc,
        model_name="sentence-transformers/all-minilm-l6-v2",
        chunk_size=500,
        chunk_overlap=50,
        save_path="vector_store"
    ):
        super().__init__(doc, model_name, chunk_size, chunk_overlap)
        self.save_path = save_path

    def run(self):
        logger.info("Автоматическая обработка документа началась")
        chunks = self.chunk()
        vector_store = self.embedder(chunks)
        self.save_vector_store(vector_store, path=self.save_path)
        logger.info("Автоматическая обработка документа завершена")
        return {"status": "success", "path": self.save_path}

        

        