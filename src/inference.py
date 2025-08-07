from langchain_core.prompts import PromptTemplate
from langchain_together import Together
from langchain_core.tools import Tool
from langchain_core.output_parsers import StrOutputParser


from src.preprocess_docs import AutomatedPreprocess
from src.retriever import RetrieverSystem
from utils.prompts import prompt_rag, prompt_llm
from utils.states import State

# Initialize the LLM with Together and output parser
llm = Together(
    model="Qwen/Qwen3-235B-A22B-Thinking-2507",
    together_api_key="8fb4509e710af87745bc5853b76aae0d79defcdf7420d3611d026060f48809df",
    temperature=0.5,
    max_tokens=300
)
output_parser = StrOutputParser()


#TOOLS:
#TOOL for creating vector store
def create_vector_store_tool(state: State) -> State:
    preprocess = AutomatedPreprocess(state.docs)
    preprocess.run()
    return state
#TOOL for retrieving information from vector store
def retrieve_information_tool(state: State) -> State:
    context = RetrieverSystem().retrieve(state.query)
    context = context or ""
    chain = prompt_rag | llm | output_parser
    answer = chain.invoke({"context": context, "question": state.query})
    return State(query=state.query, answer=answer, context=context)
#TOOL for asking LLM
def ask_llm_tool(state: State) -> State:
    chain = prompt_llm | llm | output_parser
    answer = chain.invoke({"query": state.query})
    return State(query=state.query, answer=answer)