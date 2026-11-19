"""Tests run on a tiny randomly initialised GPT-2, so they need no network."""
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from prism.backend import WhiteBoxModel


class CharTok:
    """Minimal tokenizer: one token per character (ids 1..), BOS = 0."""
    bos_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [1 + (ord(c) % 60) for c in text]

        class R: pass
        r = R()
        r.input_ids = torch.tensor([ids]) if return_tensors == "pt" else ids
        return r

    def decode(self, ids):
        return "".join("<bos>" if i == 0 else chr(96 + (int(i) % 26)) for i in ids)


@pytest.fixture(scope="session")
def model():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=3, n_head=4)
    return WhiteBoxModel(GPT2LMHeadModel(cfg), CharTok(), name="tiny")
