"""日志配置工具。

本模块是 cutie_cpp 日志的唯一配置入口，业务模块不要自行 addHandler，
否则同一条日志会被输出多次。

用法：
    程序入口调用一次 setup_logging()，业务模块用 get_logger(__name__) 取子 logger。
"""

import logging
import sys
from pathlib import Path


ROOT_LOGGER_NAME = "cutie_cpp"

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _build_formatter():
    """构造统一的日志格式化器。

    Returns:
        logging.Formatter: 含毫秒时间戳、等级、文件名行号的格式化器。
    """
    return logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)


def setup_logging(level=logging.INFO, log_dir=None, log_name="cutie_cpp"):
    """配置 cutie_cpp 的日志输出。

    在项目根 logger 上挂 StreamHandler（输出到 stdout）；若给了 log_dir，
    额外挂一个 FileHandler。两者共用同一个 Formatter，终端与文件格式一致。

    可重复调用：每次会先清空已有 handler 再挂载，避免交互式使用时日志翻倍。

    Args:
        level (int): 日志等级，如 logging.INFO、logging.DEBUG。
        log_dir (str | Path | None): 日志文件目录。为 None 时只输出到终端。
        log_name (str): 日志文件名前缀，仅在 log_dir 非空时使用。

    Returns:
        logging.Logger: 配置好的根 logger。
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)

    # 先清空，避免重复调用导致同一条日志输出多次
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = _build_formatter()

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / f"{log_name}.log", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 不向上传播到 root logger，避免与调用方自己的 basicConfig 重复输出
    logger.propagate = False
    return logger


def get_logger(name):
    """获取 cutie_cpp 命名空间下的子 logger。

    Args:
        name (str): 模块名，通常直接传 __name__。

    Returns:
        logging.Logger: 名为 cutie_cpp.<module> 的子 logger。
    """
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)

    # 把 __name__ 的包前缀归一到 cutie_cpp 命名空间下
    short_name = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{short_name}")
