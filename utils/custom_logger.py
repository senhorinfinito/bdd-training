import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("%(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


logger = get_logger(__name__)


def one_line_symbol():
    logger.info(f"{'-' * 165}")
