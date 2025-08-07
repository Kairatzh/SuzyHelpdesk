
from utils.states import State

#Роутер чтобы понять использовать ли create_vector_store для создания векторного хранилище или проигнорировать ветку
def router_cdb(state: State) -> str:
    if state.docs is not None:
        return "create_vector_store"
    else:
        return "llm"