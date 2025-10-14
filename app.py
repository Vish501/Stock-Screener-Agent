from typing import Annotated
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from colorama import Fore

# 1. Initialize LLM
llm = ChatOllama(model="qwen2.5:14b") # Q4_K_M

# 2. Create State
class State(dict):
    messages: Annotated[list, add_messages]

# 3. Build LLM Nodes
def chatbot(state: State):
    return {"messages":[llm.invoke(state["messages"])]}

# 4. Assembling Graph Structure
# 5. Adding Memory and Compiling Graph
# 6. Call Loop and Running the Loop
# 7. Create Tool
# 8. Bind LLM with Tools
# 9. Create Tool Node
# 10. Create Router Node
# 11. Update Graph for Tools
