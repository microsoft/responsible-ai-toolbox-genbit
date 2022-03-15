# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import os
import json
from parameterized import parameterized
from genbit import GenBitMetrics


class NonBinaryGenderTestCase(unittest.TestCase):

    input_sentences_non_binary_test_path = "tests/test_data_files/input/non_binary_test.txt"
    input_sentences_balanced_non_binary_test_path = "tests/test_data_files/input/balanced_binary_non_binary_test.txt"

    def testAnalyzeNonBinaryText(self):
        genbit_metrics = GenBitMetrics("en")
        input_file = os.path.join(
            os.getcwd(), self.input_sentences_non_binary_test_path)
        with open(input_file, "r", encoding="utf-8") as input_sentence_file:
            input_text = input_sentence_file.readlines()

        genbit_metrics.add_data(input_text)

        results = genbit_metrics.get_metrics(output_word_list=False)
        # print(json.dumps(results, indent=1))

        self.assertAlmostEqual(results["genbit_score"], 0.0)
        self.assertAlmostEqual(results["percentage_of_female_gender_definition_words"], 0.0)
        self.assertAlmostEqual(results["percentage_of_male_gender_definition_words"], 0.0)
        self.assertAlmostEqual(results["percentage_of_non_binary_gender_definition_words"], 1.0)

    def testAnalyzeBalancedText(self):
        genbit_metrics = GenBitMetrics("en")
        input_file = os.path.join(
            os.getcwd(), self.input_sentences_balanced_non_binary_test_path)
        with open(input_file, "r", encoding="utf-8") as input_sentence_file:
            input_text = input_sentence_file.readlines()

        genbit_metrics.add_data(input_text)

        results = genbit_metrics.get_metrics(output_word_list=False)
        # print(json.dumps(results, indent=1))

        self.assertAlmostEqual(results["genbit_score"], 0.3585693)
        self.assertAlmostEqual(results["percentage_of_female_gender_definition_words"], 0.24)
        self.assertAlmostEqual(results["percentage_of_male_gender_definition_words"], 0.3)
        self.assertAlmostEqual(results["percentage_of_non_binary_gender_definition_words"], 0.46)
