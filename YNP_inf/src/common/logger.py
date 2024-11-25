import logging

logger_name = "yolo-nas-pose"
py_logger = logging.getLogger(logger_name)
py_logger.setLevel(logging.DEBUG)

py_handler = logging.FileHandler(f"{logger_name}.log", mode='a')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

py_handler.setFormatter(py_formatter)
py_logger.addHandler(py_handler)