from app.core.constants import (
    DistributedLimits,
    JobConfigLimits,
    JobLimits,
    RunLimits,
)


def test_job_limits_are_positive_and_stable():
    assert JobLimits.NAME_MAX_LENGTH == 128
    assert JobLimits.DESCRIPTION_MAX_LENGTH == 1000
    assert JobConfigLimits.VERSION_MIN == 1


def test_run_limits_are_positive_and_stable():
    assert RunLimits.ID_MAX_LENGTH == 8
    assert RunLimits.COMMAND_MAX_LENGTH == 2000
    assert RunLimits.ERROR_MESSAGE_MAX_LENGTH == 2000
    assert RunLimits.LOG_LEVEL_MAX_LENGTH == 16
    assert RunLimits.LOG_MESSAGE_MAX_LENGTH == 8000
    assert RunLimits.ARTIFACT_TYPE_MAX_LENGTH == 64
    assert RunLimits.ARTIFACT_PATH_MAX_LENGTH == 2000


def test_distributed_limits_are_positive_and_stable():
    assert DistributedLimits.ID_MAX_LENGTH == 8
    assert DistributedLimits.NAME_MAX_LENGTH == 128
    assert DistributedLimits.DESCRIPTION_MAX_LENGTH == 1000
    assert DistributedLimits.CLIENT_NAME_MAX_LENGTH == 128
    assert DistributedLimits.SERVER_IP_MAX_LENGTH == 128
