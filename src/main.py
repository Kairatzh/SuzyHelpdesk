from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from src.agent import AgentGraph
from src.preprocess_docs import AutomatedPreprocess
from src.utils.states import State

app = FastAPI(
              title="SuzyHelpdesk API", 
              description="API for SuzyHelpdesk, AI helpdesk system",
              version="0.1.0",
              contact={
                  "name": "SuzyHelpdesk CEO",
                  "email": "zhaksylykov.k06@gmail.com"
              },
              )
agent = AgentGraph()
agent.build_graph()

class Query(BaseModel):
    question: str = Field(..., description="Этот вопрос будет обработан агентом SuzyHelpdesk")
    document: str = Field(..., description="Этот документ будет обработан агентом Suzyhelpdesk")
    user_id: str

@app.post("/document_loader", summary="Загрузка документа")
def document_loader(query: Query):
    path = f"vector_store/{query.user_id}"
    preprocess = AutomatedPreprocess(
        doc=query.document,
        save_path=path
    )
    preprocess.run()
    return {"message": f"Документ для пользователя {query.user_id} успешно обработан"}

@app.post("/query", summary="Обработка запроса", description="Этот эндпоинт принимает вопрос и ищет в векторном хранилище ответ на него")
def query(query: Query):
    new_state = State(query=query.question, user_id=query.user_id)
    return agent.run(new_state)

@app.get("/", summary="Проверка работоспособности API")
def root():
    return {"message": "SuzyHelpdesk API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")