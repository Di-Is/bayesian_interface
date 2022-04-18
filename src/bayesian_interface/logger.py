import logging

import tqdm


class Logger:
    def __init__(self, name: str, level=logging.INFO):

        logging.basicConfig(level=level)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        self.logger = logger

    def info(self, msg: str):
        self.logger.info(self.format(msg))

    def debug(self, msg: str):
        self.logger.debug(self.format(msg))

    def error(self, msg: str):
        self.logger.error(self.format(msg))

    def fatal(self, msg: str):
        self.logger.fatal(self.format(msg))

    def warning(self, msg: str):
        self.logger.warning(self.format(msg))

    def format(self, msg: str) -> str:
        return f"{msg}"


class _DummyProgressBar(object):
    """ダミーのプログレスバー"""

    def __init__(self):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, count):
        pass


def get_progress_bar(display, *arg, **kwargs):
    """プログレスバーのオブジェクトを取得する"""
    if display:
        if display is True:
            pbar = tqdm.tqdm(*arg, **kwargs)
        else:
            pbar = getattr(tqdm, "tqdm_" + display)(*arg, **kwargs)
    else:
        pbar = _DummyProgressBar()
    return pbar
