from typing import Annotated, get_args, get_origin

from fastapi.params import Depends

from app.api.deps import AsyncSessionDep
from app.core.db import get_session


def test_async_session_dep_uses_get_session_dependency():
    assert get_origin(AsyncSessionDep) is Annotated

    args = get_args(AsyncSessionDep)
    dependency = args[1]
    assert isinstance(dependency, Depends)
    assert dependency.dependency is get_session
