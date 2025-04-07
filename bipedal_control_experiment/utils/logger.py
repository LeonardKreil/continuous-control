import os
import logging
from datetime import datetime

def setup_logger(name, log_dir="./results/logs"):
    """Erstellt und konfiguriert einen Logger."""
    # Stelle sicher, dass das Log-Verzeichnis existiert
    os.makedirs(log_dir, exist_ok=True)
    
    # Erstelle Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Verhindere doppelte Handler
    if not logger.handlers:
        # Datei-Handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f"{log_dir}/{name}_{timestamp}.log")
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        # Konsolen-Handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    return logger