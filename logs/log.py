import logging

logging.basicConfig(
    level=logging.DEBUG,  # hoặc INFO, WARNING, ERROR
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Ghi ra console
        logging.FileHandler("app.log", encoding="utf-8")  # Ghi ra file
    ]
)