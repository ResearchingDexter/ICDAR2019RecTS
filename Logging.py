import logging
LOG_FORMAT="%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT="%Y/%m/%d %H:%M:%S %p"
logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,datefmt=DATE_FORMAT)