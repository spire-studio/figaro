from __future__ import annotations

from typing import Final


class JobLimits:
    NAME_MAX_LENGTH: Final[int] = 128
    DESCRIPTION_MAX_LENGTH: Final[int] = 1000


class JobConfigLimits:
    VERSION_MIN: Final[int] = 1


class RunLimits:
    ID_MAX_LENGTH: Final[int] = 8
    COMMAND_MAX_LENGTH: Final[int] = 2000
    ERROR_MESSAGE_MAX_LENGTH: Final[int] = 2000
    LOG_LEVEL_MAX_LENGTH: Final[int] = 16
    LOG_MESSAGE_MAX_LENGTH: Final[int] = 8000
    ARTIFACT_TYPE_MAX_LENGTH: Final[int] = 64
    ARTIFACT_PATH_MAX_LENGTH: Final[int] = 2000


class DistributedLimits:
    ID_MAX_LENGTH: Final[int] = 8
    NAME_MAX_LENGTH: Final[int] = 128
    DESCRIPTION_MAX_LENGTH: Final[int] = 1000
    CLIENT_NAME_MAX_LENGTH: Final[int] = 128
    SERVER_IP_MAX_LENGTH: Final[int] = 128
