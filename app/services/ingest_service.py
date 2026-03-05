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
        
    def split_law_documents(self, file_path: str):
        import re
        from langchain_core.documents import Document
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        documents = []
        current_chuong = "Nghị định 100/2019/NĐ-CP"
        current_dieu = ""
        current_khoan = ""
        current_text = []

        def flush_chunk():
            if current_text:
                text = "\n".join(current_text)
                context_str = f"Trích {current_chuong}"
                if current_dieu:
                    context_str += f", {current_dieu}"
                if current_khoan:
                    context_str += f", Khoản {current_khoan}"
                
                content = f"[{context_str}]\n{text}"
                documents.append(Document(page_content=content, metadata={"source": file_path}))
                current_text.clear()

        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Chương"):
                current_chuong = "Nghị định 100/2019/NĐ-CP - " + line
                continue
                
            dieu_match = re.match(r"^(Điều \d+\..*)$", line)
            if dieu_match:
                flush_chunk()
                current_dieu = dieu_match.group(1).split('.')[0] # E.g., 'Điều 5'
                current_khoan = ""
                current_text.append(line)
                continue
                
            khoan_match = re.match(r"^(\d+)\.\s", line)
            if khoan_match:
                flush_chunk()
                current_khoan = khoan_match.group(1)
                current_text.append(line)
                continue
                
            current_text.append(line)
            
        flush_chunk()
        return documents

    def ingest_text_file(self, file_path: str, max_chunks: int = 0, batch_size: int = 10):
        """
        Reads a text file, splits it into semantic chunks, extracts graph data, and saves to Neo4j.
        Calculates embeddings for 'HànhViViPhạm' nodes.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Custom parsing document from {file_path} to preserve context...")
        chunks = self.split_law_documents(file_path)
        logger.info(f"Split document into {len(chunks)} chunks based on Điều/Khoản.")
        
        # We can still apply a text splitter if some Khoản are abnormally large, but skip for now
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        # chunks = text_splitter.split_documents(chunks)
        
        if max_chunks is not None and max_chunks > 0:
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
                
        # Generate Embeddings for 'HànhViViPhạm' Nodes
        self.calculate_embeddings()
        
        # Refresh the schema to make the new elements available for Cypher QA chains
        self.graph.refresh_schema()
        logger.info("Ingestion completed and schema refreshed successfully.")

    def calculate_embeddings(self):
        from langchain_openai import OpenAIEmbeddings
        logger.info("Calculating embeddings for Hànhviviphạm nodes...")
        embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        
        try:
            # Tạo Vector Index trên Hànhviviphạm
            self.graph.query("CREATE VECTOR INDEX hành_vi_index IF NOT EXISTS FOR (n:Hànhviviphạm) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}")
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")

        # Lấy các node Hànhviviphạm chưa có embedding
        nodes = self.graph.query("MATCH (n:Hànhviviphạm) WHERE n.embedding IS NULL RETURN elementId(n) AS id, n.id AS text")
        
        if not nodes:
            logger.info("No new nodes strictly require embedding updates.")
            return

        logger.info(f"Found {len(nodes)} Hànhviviphạm nodes to embed.")
        for node in nodes:
            try:
                vec = embeddings.embed_query(node["text"])
                self.graph.query(
                    "MATCH (n) WHERE elementId(n) = $id CALL db.create.setNodeVectorProperty(n, 'embedding', $vec) RETURN count(*)", 
                    params={"id": node["id"], "vec": vec}
                )
            except Exception as e:
                logger.error(f"Failed to set embedding for node {node['id']}: {e}")
        logger.info("Finished embedding calculation.")
        
ingest_service = IngestService()
