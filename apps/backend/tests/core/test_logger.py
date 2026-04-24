import logging

from app.core.logger import InterceptHandler, get_logger, setup_logging


def test_get_logger_root_and_child():
    root_logger = get_logger()
    child_logger = get_logger("service.auth")

    assert root_logger.name == "app"
    assert child_logger.name == "app.service.auth"


def test_setup_logging_configures_uvicorn_loggers():
    setup_logging()

    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(logger_name)
        assert logger.handlers
        assert any(isinstance(handler, InterceptHandler) for handler in logger.handlers)
        assert logger.propagate is False
