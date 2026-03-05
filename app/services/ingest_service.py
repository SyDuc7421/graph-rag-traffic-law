import os
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from app.core.database import get_neo4j_graph
from app.core.config import settings

logger = logging.getLogger(__name__)

class IngestService:
    def __init__(self):
        self.graph = get_neo4j_graph()
        self.llm = ChatOpenAI(
            model="gpt-5-nano",
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )
        # LLMGraphTransformer uses the LLM to extract graph nodes and edges
        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            # Define the allowed nodes and relationships for the LLM to extract
            allowed_nodes=["Luật", "ĐiềuKhoản", "HànhViViPhạm", "PhươngTiện", "XửPhạt", "BiệnPháp", "ĐốiTượng"],
            allowed_relationships=["QUY_ĐỊNH", "ÁP_DỤNG_CHO", "XỬ_PHẠT_BẰNG", "BAO_GỒM", "MỨC_PHẠT", "NGOẠI_TRỪ"]
        )
        
    def ingest_text_file(self, file_path: str, max_chunks: int = 0, batch_size: int = 10):
        """
        Reads a text file, splits it into chunks, extracts graph data, and saves to Neo4j.
        Supports batching and a limit on chunks for testing.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Loading document from {file_path}")
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        # Split text into manageable chunks
        # Chunk size is kept relatively small to not overwhelm the LLM context and get precise entities
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
            logger.info(f"Limiting to first {len(chunks)} chunks for testing.")
            
        total_chunks = len(chunks)
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} (chunks {i + 1} to {min(i + batch_size, total_chunks)} of {total_chunks})...")
            
            try:
                # Extract graph data
                graph_documents = self.llm_transformer.convert_to_graph_documents(batch_chunks)
                
                # Ingest to Neo4j
                logger.info(f"Extracted {len(graph_documents)} graph documents. Adding to Neo4j...")
                self.graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
            except Exception as e:
                logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                
        # Refresh the schema to make the new elements available for Cypher QA chains
        self.graph.refresh_schema()
        logger.info("Ingestion completed and schema refreshed successfully.")
        
ingest_service = IngestService()
