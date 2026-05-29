import logging
import sys

def setup_logging(level: str = "INFO"):
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    logger.addHandler(handler)
    
    # Убираем дубли от uvicorn
    logging.getLogger("uvicorn").handlers = []