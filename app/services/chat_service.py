from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from app.core.database import get_neo4j_graph
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class QueryIntent(BaseModel):
    intent: str = Field(description="Mục đích của câu hỏi, ví dụ: hỏi mức phạt, hỏi hình thức bổ sung, hỏi đối tượng")
    subject: str = Field(description="Đối tượng/phương tiện điều khiển bị nhắc đến (ví dụ: xe máy, ô tô, xe đạp, đi bộ...)", default="")
    behavior_keywords: str = Field(description="Cụm từ tiếng Việt rõ nghĩa mô tả hành vi vi phạm (ví dụ: vượt đèn đỏ, say xỉn, không đội mũ bảo hiểm)", default="")

class ChatService:
    def __init__(self):
        self.graph = get_neo4j_graph()
        self.llm = ChatOpenAI(
            model="gpt-5-nano", 
            temperature=0, 
            api_key=settings.OPENAI_API_KEY
        )
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        
        # Pydantic schema for LLM structured output
        self.intent_parser = self.llm.with_structured_output(QueryIntent)

    def extract_intent(self, question: str) -> QueryIntent:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là chuyên gia phân tích ngôn ngữ tự nhiên. Trích xuất ý định, đối tượng phương tiện và từ khóa hành vi từ câu hỏi liên quan đến luật giao thông của người dùng."),
            ("human", "{question}")
        ])
        chain = prompt | self.intent_parser
        try:
            return chain.invoke({"question": question})
        except Exception as e:
            logger.error(f"Error parsing intent: {e}")
            return QueryIntent(intent="unknown", subject="", behavior_keywords=question)

    def vector_search_graph(self, intent: QueryIntent) -> list:
        # Nếu không có behavior, tìm thẳng trong DB hoặc trả rỗng, ở đây ta ưu tiên lấy embeddings của behavior hoặc cả câu
        search_query = intent.behavior_keywords if intent.behavior_keywords else intent.intent
        
        # Tính toán embedding
        query_vector = self.embeddings.embed_query(search_query)

        # Viết Cypher: Tìm kiếm Vector -> Lọc theo Subject (nếu có) -> Lấy dữ liệu 1-hop
        subject_filter = f"WHERE toLower(subject.id) CONTAINS toLower('{intent.subject}')" if intent.subject else ""
        
        cypher = f"""
        CALL db.index.vector.queryNodes('hành_vi_index', 3, $query_vector)
        YIELD node AS behavior, score
        WHERE score > 0.85 
        MATCH (behavior)
        OPTIONAL MATCH (behavior)-[r]-(related)
        // If subject is queried, prioritize nodes that connect to a Subject matching user query
        OPTIONAL MATCH (behavior)-[:ÁP_DỤNG_CHO]->(subject:Đốitượng)
        {subject_filter}
        RETURN behavior.id AS hanh_vi, score, type(r) AS moi_quan_he, related.id AS thong_tin_lien_quan, 
               related.id AS name, labels(related) AS label
        ORDER BY score DESC
        """
        
        try:
            results = self.graph.query(cypher, params={"query_vector": query_vector})
            return results
        except Exception as e:
            logger.error(f"Vector search failed (Index may not exist): {e}")
            return []

    def format_context(self, search_results: list) -> str:
        if not search_results:
            return ""
            
        context_lines = []
        for row in search_results:
            context_lines.append(f"Hành vi: '{row['hanh_vi']}' -> [{row['moi_quan_he']}] -> {row['label'][0] if row['label'] else ''}: '{row['thong_tin_lien_quan']}'")
        
        # Deduplicate
        return "\n".join(list(set(context_lines)))
        
    def ask(self, question: str) -> dict:
        try:
            # 1. Trích xuất Entity & Intent
            intent_data = self.extract_intent(question)
            logger.info(f"Extracted Intent: {intent_data.model_dump()}")
            
            # 2. Semantic Search trên Neo4j
            graph_results = self.vector_search_graph(intent_data)
            context_text = self.format_context(graph_results)
            logger.info(f"Retrieved Graph Context:\n{context_text}")
            
            # 3. Prompt cực kỳ nghiêm ngặt chống Hallucination
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "Bạn là luật sư giao thông đường bộ Việt Nam. "
                 "Dưới đây là căn cứ pháp lý được trích xuất từ cơ sở dữ liệu dựa trên câu hỏi của người dùng:\n"
                 "--- BẮT ĐẦU CĂN CỨ ---\n{context}\n--- KẾT THÚC CĂN CỨ ---\n\n"
                 "Quy tắc tối thượng: BẠN CHỈ ĐƯỢC PHÉP TRẢ LỜI DỰA TRÊN CĂN CỨ BÊN TRÊN MÀ KHÔNG DÙNG KIẾN THỨC CÁ NHÂN. "
                 "Nếu CĂN CỨ RỖNG hoặc không chứa thông tin trả lời cho câu hỏi, BẮT BUỘC bạn phải trả lời chính xác câu sau: 'Không biết / Không có dữ liệu'. "
                 "Nếu có đủ thông tin, hãy trình bày một cách dễ hiểu, MẠCH LẠC, rõ ràng mức phạt áp dụng cho ai và BẮT BUỘC liệt kê Trích dẫn (Quy định tại ...)."
                ),
                ("human", "{question}")
            ])
            
            final_chain = final_prompt | self.llm
            response = final_chain.invoke({
                "context": context_text,
                "question": question
            })
            
            return {
                "answer": response.content,
                "data": {
                    "extracted_intent": intent_data.model_dump(),
                    "retrieved_nodes": graph_results
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ChatService.ask: {e}")
            raise e

# Create a common singleton instance
chat_service = ChatService()
