from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from src.inference import create_vector_store_tool, retrieve_information_tool, ask_llm_tool #TOOLS
from src.routers import router_cdb #ROUTERS
from utils.states import State #STATES
from utils.logging import logger #LOGS

# Класс для создания и управления графом агентов
# Этот класс инкапсулирует логику создания графа, добавления узлов и ребер, а также запуска графа
# и визуализации его структуры
class AgentGraph:
    def __init__(self):
        self.graph = self.build_graph()

    def build_graph(self):
        graph = StateGraph(State)
        graph.add_node("start", RunnableLambda(lambda state: state))
        graph.add_node("create_vector_store", create_vector_store_tool)
        graph.add_node("retrieve_information", retrieve_information_tool)
        graph.add_node("llm", ask_llm_tool)

        graph.set_entry_point("start")

        graph.add_conditional_edges(
            "start",
            router_cdb,
            {
                "create_vector_store": "create_vector_store",
                "llm": "llm"
            }
        )

        graph.add_edge("create_vector_store", "retrieve_information")
        graph.add_edge("retrieve_information", END)
        graph.add_edge("llm", END)

        return graph.compile()

    def run(self, state: State):
        return self.graph.invoke(state)

    def draw_graph(self, name_photo: str = "graph.png") -> None:
        with open(name_photo, "wb") as f:
            f.write(self.graph.get_graph().draw_mermaid_png())

# Пример использования класса 
agent_graph = AgentGraph()
logger.info("Агентский граф успешно создан")

# Визуализация графа
agent_graph.draw_graph()
logger.info("Граф агентов успешно визуализирован и сохранен в файл")

if __name__ == "__main__":
    state = State(query="Что такое хромосома?", user_id="111")
    result = agent_graph.run(state=state)
    print(result)
    logger.info(f"Результат выполнения графа: {result}")