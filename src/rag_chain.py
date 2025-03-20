from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.graph import END, MessagesState
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated,List, TypedDict

class Search(TypedDict):
    """Search query."""
    query: Annotated[str, ..., "Search query to run."]
   


class State(TypedDict):
    question: str
    query: str
    context: List[Document]
    answer: str

class RAGChain:
    """Classe para configurar e executar a cadeia RAG."""
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.memory = MemorySaver()

    def analyze_query(state: State):
        # structured_llm = llm.with_structured_output(Search)
        # query = structured_llm.invoke(state["question"])
        return {"query": state["question"]}

    @tool(response_format="content_and_artifact")
    def retrieve(self, state: State):
        """
        Recupera documentos relevantes do vector store baseado na query do usuário.
        
        Args:
            state: Estado atual contendo a query do usuário
            
        Returns:
            Uma tupla contendo o conteúdo serializado e os documentos recuperados
        """
        query = state["query"]
        # filter_dict = {"section": query["section"]}
        retrieved_docs = self.vector_store.similarity_search(
            query,
            k=30
        )
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs


    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(self, state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = self.llm.bind_tools([self.retrieve])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}


    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieve])


    # Step 3: Generate a response using the retrieved content.
    def generate(self, state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    def create_agent(self):
        agent_executor = create_react_agent(self.llm, [self.retrieve], checkpointer=self.memory)
        return agent_executor

    def create_graph(self):
        graph_builder = StateGraph(MessagesState)

        graph_builder.add_node(self.query_or_respond)
        graph_builder.add_node(self.tools)
        graph_builder.add_node(self.generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        
        # config = {"configurable": {"thread_id": "abc123"}}

        graph = graph_builder.compile(checkpointer=self.memory)
        return graph

