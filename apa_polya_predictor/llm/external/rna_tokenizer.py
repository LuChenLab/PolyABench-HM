from __future__ import annotations

import collections
import os

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class RNATokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find vocab file: {vocab_file}")
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict((ids, tok) for tok, ids in self.vocab.items())
        self.basic_tokenizer = BasicTokenizer(do_lower_case=False, never_split=never_split)
        super().__init__(
            do_lower_case=False,
            do_basic_tokenize=True,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, split_special_tokens=False):
        return self.basic_tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens).replace(" ##", "").strip()
