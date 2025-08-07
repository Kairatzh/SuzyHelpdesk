from langchain_core.prompts import PromptTemplate

#################### PROMPT FOR RAG
# Это шаблон для RAG, Который используется для обработки вопросов и генерации ответов на основе контекста
prompt_rag_template = "Ты ассистент который читает документ и отвечает на вопросы по нему.Даже если вопрос не связан с документом, ты должен ответить на него, исходя из прочитанного. Если ты не можешь ответить на вопрос, скажи, что не знаешь ответа. Документ: {context}, Вопрос: {question}, Твой ответ:"
prompt_template_rag = prompt_rag_template
prompt_rag = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template_rag
)

##################### PROMPT FOR LLM
# Это шаблон для LLM, который используется для обработки вопросов и генерации ответов
prompt_llm_template = "Ты ассистент, который отвечает на вопросы.Если ты не можешь ответить на вопрос, скажи, что не знаешь ответа. Вопрос: {query}, Твой ответ:"
prompt_template_llm = prompt_llm_template
prompt_llm = PromptTemplate(
    input_variables=["question"],
    template=prompt_template_llm
)