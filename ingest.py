import sys
import logging
from app.services.ingest_service import ingest_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    file_path = "law.txt"
    max_chunks = None
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            max_chunks = int(sys.argv[2])
        except ValueError:
            logger.warning("Invalid max_chunks value, ignoring.")
        
    try:
        logger.info(f"Starting ingestion for {file_path}")
        ingest_service.ingest_text_file(file_path, max_chunks=max_chunks, batch_size=5)
        logger.info("Done!")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
