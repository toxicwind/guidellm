from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from guidellm.data.tokenizers.tokenizer import DataTokenizer, TokenizerRegistry
from guidellm.schemas.data.tokenizers import HuggingFaceTokenizerArgs

__all__ = ["HuggingFaceTokenizer"]


def _classify_load_error(model: str, err: Exception) -> str:
    """Turn an AutoTokenizer.from_pretrained failure into an actionable message.

    Borrowed from the failure taxonomy in vllm-project/guidellm#205 (open):
    invalid tokenizer names surface as inconsistent, non-descriptive errors.
    We classify the common Hub failure modes so the operator knows exactly
    what to fix instead of guessing.

    Matching is message-first, type-second: transformers frequently wraps
    Hub errors in a builtin OSError that keeps only the message text, so
    type checks alone miss the most common real-world failures.
    """
    cls = type(err)
    module = getattr(cls, "__module__", "") or ""
    name = cls.__name__
    msg = str(err)
    low = msg.lower()

    def hub_error(*names: str) -> bool:
        return "huggingface_hub" in module and name in names

    invalid_id = hub_error("HFValidationError") or (
        "repo id must use alphanumeric" in low or "invalid repo id" in low
    )
    if invalid_id:
        return (
            f"{model!r} is not a valid Hugging Face repo id "
            "(repo ids look like 'org/name'; characters such as ':' are forbidden). "
            "This usually means a served-model alias was used as the tokenizer. "
            "Pass the real tokenizer repo explicitly, e.g. "
            "--tokenizer kind=huggingface_auto,model=<org>/<repo>."
        )
    gated = hub_error("GatedRepoError") or "gated repo" in low
    if gated:
        return (
            f"{model!r} is a gated repo. Accept its terms on huggingface.co, then "
            "authenticate (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)."
        )
    not_found = hub_error("RepositoryNotFoundError", "EntryNotFoundError") or (
        "repository not found" in low
    )
    if not_found:
        return (
            f"Hugging Face repo {model!r} was not found. Check the repo id for typos, "
            "or pass an explicit tokenizer repo, e.g. "
            "--tokenizer kind=huggingface_auto,model=<org>/<repo>."
        )
    status = getattr(getattr(err, "response", None), "status_code", None)
    if status == 401 or "401 client error" in low:
        return (
            f"Unauthorized (HTTP 401) loading the tokenizer for {model!r}. "
            "The repo may be gated or private: accept its terms and set "
            "HF_TOKEN or HUGGING_FACE_HUB_TOKEN."
        )
    if status == 404 or "404 client error" in low:
        return (
            f"Tokenizer files for {model!r} were not found (HTTP 404). "
            "Check the repo id for typos, or pass an explicit tokenizer repo, e.g. "
            "--tokenizer kind=huggingface_auto,model=<org>/<repo>."
        )
    return f"Failed to load tokenizer for {model!r}: {err}"


@TokenizerRegistry.register(["huggingface_auto", "hf_auto"])
class HuggingFaceTokenizer(DataTokenizer):
    """Tokenizer for Hugging Face models."""

    def __init__(
        self,
        config: HuggingFaceTokenizerArgs,
    ) -> None:
        if config.model is None:
            raise ValueError("The 'name' field must be provided")

        self._config = config
        self._tokenizer: None | PreTrainedTokenizerBase = None

    def __call__(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from_pretrained = AutoTokenizer.from_pretrained(
                self._config.model,
                **self._config.load_kwargs,
            )
        except Exception as err:
            raise ValueError(_classify_load_error(self._config.model, err)) from err
        self._tokenizer = from_pretrained
        return from_pretrained
