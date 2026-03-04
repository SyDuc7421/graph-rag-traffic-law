import os
from langchain_neo4j import Neo4jGraph
from app.core.config import settings

def get_neo4j_graph() -> Neo4jGraph:
    """
    Returns an instance of Neo4jGraph connected to the database.
    LangChain will automatically inspect the schema.
    """
    # Langchain Neo4jGraph reads from env directly, but we explicitly pass settings to be safe
    os.environ["NEO4J_URI"] = settings.NEO4J_URI
    os.environ["NEO4J_USERNAME"] = settings.NEO4J_USER
    os.environ["NEO4J_PASSWORD"] = settings.NEO4J_PASSWORD
    
    graph = Neo4jGraph(
        url=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )
    return graph
