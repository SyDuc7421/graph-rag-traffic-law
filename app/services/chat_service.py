from langchain_neo4j import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from app.core.database import get_neo4j_graph
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        # Database connection setup
        self.graph = get_neo4j_graph()
        
        # LLM (OpenAI) configuration
        self.llm = ChatOpenAI(
            model="gpt-5-nano", 
            temperature=0, 
            api_key=settings.OPENAI_API_KEY
        )
        
        # Build QA Chain: The bot uses LLM + Graph Schema to generate Cypher queries
        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            cypher_llm=self.llm,   # Model to construct Cypher Query
            qa_llm=self.llm,       # Model to synthesize the natural response
            verbose=True,
            return_direct=False,   # Return natural language response instead of pure query result
            allow_dangerous_requests=True
        )

    def ask(self, question: str) -> dict:
        """
        Sends query to LangChain QA Chain
        """
        try:
            # Refresh schema if the database undergoes any updates
            self.graph.refresh_schema()
            
            # Execute chain
            response = self.chain.invoke({"query": question})
            
            return {
                "answer": response.get("result"),
                # Return context data if it's necessary for debugging later
                "data": None
            }
        except Exception as e:
            logger.error(f"Error in ChatService.ask: {e}")
            raise e

# Create a common singleton instance
chat_service = ChatService()
