"""
    rag_pipeline.py
    Этот модуль содержит класс RetieverSystem, который отвечает за извлечение информации из векторного хранилище
"""
# Импорт нужных библиотек
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from utils.logging import logger

#Создание класса для извлечения информации из векторного хранилище
class RetrieverSystem:
    def __init__(self, save_path):
        self.save_path = save_path
        self.retiever = ContextualCompressionRetriever()
    def retrieve(self, query):
        logger.info("Начало извлечения информации")
        try:
            vector_store = FAISS.load_local(self.save_path)
            self.retiever = vector_store.as_retriever()
            results = self.retriever.get_relevant_documents(query)
            logger.info("Извлечение информации завершено")
            return results
        except Exception as e:
            logger.error(f"Ошибка при извлечении информации: {e}")



