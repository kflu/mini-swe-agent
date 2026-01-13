import logging
from collections.abc import Callable

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.openai_utils import coerce_responses_text

logger = logging.getLogger("litellm_response_api_model")


class LitellmResponseAPIModelConfig(LitellmModelConfig):
    pass


class LitellmResponseAPIModel(LitellmModel):
    def __init__(self, *, config_class: Callable = LitellmResponseAPIModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    @retry(
        reraise=True,
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            # Remove 'timestamp' field added by agent - not supported by OpenAI responses API
            clean_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            resp = litellm.responses(
                model=self.config.model_name,
                input=clean_messages,
                previous_response_id=None,
                **(self.config.model_kwargs | kwargs),
            )
            return resp
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)
        text = coerce_responses_text(response)
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        return {
            "content": text,
        }
