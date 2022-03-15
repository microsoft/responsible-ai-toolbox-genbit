# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import os
import json
from genbit.tokenizer import Tokenizer


class TokenizerTestCase(unittest.TestCase):

    input_sentences_path = "tests/test_data_files/input/winogender_en_50.tsv"
    expected_tokenized_text_path = "tests/test_data_files/expected/winogender_en_50_tokenized.json"
    language_code = "en"

    def testInit(self):
        tokenizer = Tokenizer(self.language_code)
        self.assertIsInstance(tokenizer, Tokenizer)
        self.assertTrue(hasattr(tokenizer, "_language_code"))
        self.assertIsNone(tokenizer._tokenizer)

    def testTokenizeData(self):
        input_file = os.path.join(os.getcwd(), self.input_sentences_path)
        with open(input_file, "r", encoding="utf-8") as input_sentence_file:
            input_text = input_sentence_file.readlines()

        tokenizer = Tokenizer(self.language_code)
        tokenized_text = tokenizer.tokenize_data(input_text)

        expected_tokenized_path = os.path.join(
            os.getcwd(), self.expected_tokenized_text_path)
        with open(expected_tokenized_path, "r", encoding="utf-8") as expected_tokenized_path:
            expected_tokenized_text = json.load(expected_tokenized_path)
        self.assertEqual(tokenized_text, expected_tokenized_text)
