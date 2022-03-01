# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from unittest.mock import patch
import os
import json
from parameterized import parameterized
from genbit import GenBitMetrics


class TokenizerTestCase(unittest.TestCase):

    input_sentences_path = "tests/test_data_files/input/winogender_en_50.tsv"
    expected_result_structure_path = "tests/test_data_files/expected/result_structure.json"

    def setUp(self):
        expected_result_path = os.path.join(
                os.getcwd(),
                self.expected_result_structure_path)

        with open(expected_result_path, "r", encoding="utf-8") as expected_result_structure_file:
            self.expected_result_structure = json.load(
                expected_result_structure_file)

    @parameterized.expand(
        [
            ["en"],
            ["es"],
            ["fr"],
            ["it"],
            ["de"],
            ["ru"],
        ])
    @patch('os.path.isfile')
    def testAnalyzeText(self, language_code, mock_isfile):
        # mock out loading non-binary.txt from this test
        mock_isfile.return_value = False

        genbit_metrics = GenBitMetrics(language_code)
        input_file = os.path.join(
            os.getcwd(), self.input_sentences_path)
        with open(input_file, "r", encoding="utf-8") as input_sentence_file:
            input_text = input_sentence_file.readlines()

        genbit_metrics.add_data(input_text)

        results = genbit_metrics.get_metrics()
        for key, value in self.expected_result_structure.items():
            self.assertIn(key, results)
            self.assertIsInstance(results[key], type(value))
