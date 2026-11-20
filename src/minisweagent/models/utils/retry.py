"""Retry utility for model queries."""

import logging
import os

from tenacity import Retrying, before_sleep_log, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from minisweagent.exceptions import UserInterruption


def retry(*, logger: logging.Logger, abort_exceptions: list[type[Exception]]) -> Retrying:
    """Thin wrapper around tenacity.Retrying to make use of global config etc.

    Args:
        logger: Logger to use for reporting retries
        abort_exceptions: Exceptions to abort on.

    Returns:
        A tenacity.Retrying object.
    """
    # A user stop must never be treated as a transient model failure. In
    # particular, the GUI raises ``UserInterruption`` from the streaming
    # callback while a request is in flight; retrying that would silently keep
    # querying the model and make the Stop button appear to do nothing.
    abort_exception_types = (UserInterruption, *abort_exceptions)
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(abort_exception_types),
    )
