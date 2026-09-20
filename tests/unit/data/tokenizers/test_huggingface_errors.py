"""Unit tests for classified tokenizer load errors in
guidellm.data.tokenizers.huggingface.

Borrowed failure taxonomy: vllm-project/guidellm#205 (open) -- invalid
tokenizer names surface as inconsistent, non-descriptive errors. Our fork
classifies the Hub failure modes into actionable messages.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    RepositoryNotFoundError,
)
from requests import Response
from requests.exceptions import HTTPError

from guidellm.data.tokenizers.huggingface import HuggingFaceTokenizer
from guidellm.schemas.data import HuggingFaceTokenizerArgs


def _hub_error(cls, status_code):
    """Build a real huggingface_hub HTTP error with a stubbed response."""
    resp = Response()
    resp.status_code = status_code
    resp.url = "https://huggingface.co/org/repo/resolve/main/tokenizer_config.json"
    return cls("hub error", response=resp)


def _tokenizer(model="qwen3:4b"):
    return HuggingFaceTokenizer(HuggingFaceTokenizerArgs(model=model))


def _fail_with(monkeypatch: pytest.MonkeyPatch, exc: Exception):
    def boom(model, **kwargs):
        raise exc

    monkeypatch.setattr(
        "guidellm.data.tokenizers.huggingface.AutoTokenizer.from_pretrained",
        boom,
    )


@pytest.mark.sanity
def test_invalid_repo_id_suggests_explicit_tokenizer(monkeypatch):
    """'qwen3:4b'-style names get an actionable message, not a raw traceback."""
    _fail_with(monkeypatch, HFValidationError("bad repo id"))
    with pytest.raises(ValueError, match="not a valid Hugging Face repo id"):
        _tokenizer("qwen3:4b")()
    try:
        _tokenizer("qwen3:4b")()
    except ValueError as err:
        assert "--tokenizer kind=huggingface_auto,model=" in str(err)


@pytest.mark.sanity
def test_not_found_suggests_checking_repo(monkeypatch):
    _fail_with(monkeypatch, _hub_error(RepositoryNotFoundError, 404))
    with pytest.raises(ValueError, match="was not found"):
        _tokenizer("org/no-such-repo")()


@pytest.mark.sanity
def test_gated_repo_suggests_auth(monkeypatch):
    _fail_with(monkeypatch, _hub_error(GatedRepoError, 401))
    with pytest.raises(ValueError, match="gated repo") as excinfo:
        _tokenizer("org/gated-repo")()
    assert "HF_TOKEN" in str(excinfo.value)


@pytest.mark.sanity
def test_http_401_suggests_auth(monkeypatch):
    err = HTTPError("401 Client Error")
    err.response = SimpleNamespace(status_code=401)
    _fail_with(monkeypatch, err)
    with pytest.raises(ValueError, match="Unauthorized.*HTTP 401"):
        _tokenizer("org/private-repo")()


@pytest.mark.sanity
def test_http_404_suggests_checking_repo(monkeypatch):
    err = HTTPError("404 Client Error")
    err.response = SimpleNamespace(status_code=404)
    _fail_with(monkeypatch, err)
    with pytest.raises(ValueError, match="HTTP 404"):
        _tokenizer("org/no-such-repo")()


@pytest.mark.sanity
def test_oserror_wrapped_invalid_id_is_classified(monkeypatch):
    """Real-world path: transformers wraps HFValidationError in builtin OSError."""
    _fail_with(
        monkeypatch,
        OSError(
            "Repo id must use alphanumeric chars, '-', '_' or '.'. "
            "The name cannot start or end with '-' or '.' "
            "and the maximum length is 96: 'qwen3:4b'."
        ),
    )
    with pytest.raises(ValueError, match="not a valid Hugging Face repo id"):
        _tokenizer("qwen3:4b")()


@pytest.mark.sanity
def test_oserror_wrapped_not_found_is_classified(monkeypatch):
    _fail_with(
        monkeypatch,
        OSError("Repository Not Found for url: https://huggingface.co/org/nope"),
    )
    with pytest.raises(ValueError, match="was not found"):
        _tokenizer("org/nope")()


@pytest.mark.sanity
def test_unknown_error_chains_original(monkeypatch):
    """Unclassified failures keep the original exception as __cause__."""
    cause = RuntimeError("boom")
    _fail_with(monkeypatch, cause)
    with pytest.raises(ValueError, match="Failed to load tokenizer") as excinfo:
        _tokenizer("gpt2")()
    assert excinfo.value.__cause__ is cause


@pytest.mark.sanity
def test_successful_load_unaffected(monkeypatch):
    """The happy path still loads and caches (no behavior change)."""
    sentinel = object()

    def fake(model, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "guidellm.data.tokenizers.huggingface.AutoTokenizer.from_pretrained",
        fake,
    )
    tok = _tokenizer("gpt2")
    assert tok() is sentinel
    assert tok() is sentinel
