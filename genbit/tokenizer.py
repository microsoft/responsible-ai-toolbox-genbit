# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tokenization for the gender bias measure. All sentences are concatenated to retain
contextual information present in the corpus"""

from typing import List
from nltk import RegexpTokenizer


class Tokenizer:
    """Tokenizer class, the tokenizer is only initialized when tokenization is needed to avoid initializing models
    before invocation"""

    def __init__(self, language_code):
        self._language_code = language_code
        self._tokenizer = None

    def tokenize_data(self, text_data:List[str]) -> List[List[str]]:
        if not self._tokenizer:
            self.initialize_tokenizer()
        tokenized_data = []
        for data_item in text_data:
            tokenized_text = self._tokenizer.tokenize(data_item)
            tokenized_data.append(tokenized_text)
        return tokenized_data

    def initialize_tokenizer(self):
        self._tokenizer = RegexpTokenizer(r'[^\W\d_]+')
