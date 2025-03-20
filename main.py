from src.document_loader import DocumentLoader
from src.embedding_manager import EmbeddingManager
from src.rag_chain import RAGChain

from src.utils import load_env_vars

load_env_vars()

all_splits = DocumentLoader.load_web("https://nearx.com.br")

embedding_manager = EmbeddingManager()
vector_store = embedding_manager.get_or_create_vector_store(all_splits)

config = {"configurable": {"thread_id": "abc123"}}

agent_executor = RAGChain(vector_store).create_agent()

while True:

    input_message = input("Digite sua pergunta: ")

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        
        last_step = step

    # Mostrar apenas o último step
    if last_step:
        print(last_step["messages"][-1].__dict__["content"])


 
    
    
    




