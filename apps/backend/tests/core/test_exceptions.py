from app.core import exceptions


def test_app_error_preserves_message_and_context_copy():
    source_context = {"job_id": 7}
    err = exceptions.AppError("boom", context=source_context)
    source_context["job_id"] = 8

    assert str(err) == "boom"
    assert err.context == {"job_id": 7}


def test_exception_hierarchy_is_consistent():
    assert issubclass(exceptions.InternalServiceError, exceptions.AppError)
    assert issubclass(exceptions.BadRequestError, exceptions.AppError)
    assert issubclass(exceptions.ResourceNotFound, exceptions.AppError)
    assert issubclass(exceptions.ResourceConflict, exceptions.AppError)
    assert issubclass(exceptions.JobNotFound, exceptions.ResourceNotFound)
    assert issubclass(exceptions.RunNotFound, exceptions.ResourceNotFound)
    assert issubclass(exceptions.JobAlreadyExists, exceptions.ResourceConflict)
    assert issubclass(exceptions.ActiveRunsConflict, exceptions.ResourceConflict)
    assert issubclass(exceptions.ActiveRunConflict, exceptions.ResourceConflict)


def test_exception_exports_include_public_error_types():
    exported = set(exceptions.__all__)
    required = {
        "AppError",
        "InternalServiceError",
        "BadRequestError",
        "ResourceNotFound",
        "ResourceConflict",
        "JobNotFound",
        "JobAlreadyExists",
        "JobConfigNotFound",
        "RunNotFound",
        "ConfigSchemaNotFound",
        "ActiveRunsConflict",
        "ActiveRunConflict",
    }
    assert required.issubset(exported)
