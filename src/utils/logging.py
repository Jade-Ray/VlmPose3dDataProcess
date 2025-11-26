import sys
import logging

LOG_FORMAT = '%(asctime)s  %(levelname)5s  %(message)s'


def get_logger(name: str, log_level: str = 'info', log_format: str = None, log_file: str = None) -> logging.Logger:
    if log_format is None:
        log_format = LOG_FORMAT
    logging.basicConfig(level=log_level.upper(), format=log_format)
    logger = logging.getLogger(name)
    logger.addHandler(get_console_handler(log_level, log_format))
    if log_file is not None:
        logger.addHandler(get_file_handler(log_file, log_level, log_format))
    logger.propagate = False # otherwise root logger prints things again
    return logger


def get_file_handler(log_file: str, log_level: str = 'info', log_format: str = None)-> logging.FileHandler:
    if log_format is None:
        log_format = LOG_FORMAT
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level.upper())
    file_handler.setFormatter(logging.Formatter(log_format))
    return file_handler


def get_console_handler(log_level: str = 'info', log_format: str = None)-> logging.StreamHandler:
    if log_format is None:
        log_format = LOG_FORMAT
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level.upper())
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Fix UTF-8 encoding for Windows console
    if sys.platform == "win32":
        # Reconfigure stdout to use UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        else:
            # Fallback: wrap with UTF-8 writer
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    
    return console_handler