# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import os
import json
from parameterized import parameterized
from genbit import GenBitMetrics


class TransCisGenderTestCase(unittest.TestCase):

    input_sentences_trans_cis_test_path = "tests/test_data_files/input/trans_cis_test.txt"
    input_sentences_balanced_trans_cis_test_path = "tests/test_data_files/input/balanced_trans_cis_test.txt"

    def testAnalyzeTransCisText(self):
        genbit_metrics = GenBitMetrics("en")

        input_file = os.path.join(
            os.getcwd(), self.input_sentences_trans_cis_test_path)
        with open(input_file, "r", encoding="utf-8") as input_sentence_file:
            input_text = input_sentence_file.readlines()

        genbit_metrics.add_data(input_text)
        results = genbit_metrics.get_metrics(output_word_list=False)

        self.assertAlmostEqual(results["genbit_score"], 0.0)
        self.assertAlmostEqual(results["percentage_of_female_gender_definition_words"], 1/3)
        self.assertAlmostEqual(results["percentage_of_male_gender_definition_words"], 1/3)
        self.assertAlmostEqual(results["percentage_of_trans_gender_definition_words"], 0.6)
        self.assertAlmostEqual(results["percentage_of_cis_gender_definition_words"], 0.4)
        self.assertAlmostEqual(results["additional_metrics"]["avg_trans_cis_bias_ratio"], -1.548421887461201)
        self.assertAlmostEqual(results["additional_metrics"]["avg_trans_cis_bias_conditional"], -1.3813678027980347)
        self.assertAlmostEqual(results["additional_metrics"]["avg_trans_cis_bias_ratio_absolute"], 1.548421887461201)
        self.assertAlmostEqual(results["additional_metrics"]["avg_trans_cis_bias_conditional_absolute"], 1.3813678027980347)

    def testAnalyzeBalancedTransCisText(self):
        genbit_metrics = GenBitMetrics("en", percentile_cutoff=0)
        input_file = os.path.join(
            os.getcwd(), self.input_sentences_balanced_trans_cis_test_path)
        with open(input_file, "r", encoding="utf-8") as input_sentence_file:
            input_text = input_sentence_file.readlines()

        genbit_metrics.add_data(input_text)

        results = genbit_metrics.get_metrics(output_word_list=False)

        self.assertAlmostEqual(results['genbit_score'], 0.0)
        self.assertAlmostEqual(results["percentage_of_trans_gender_definition_words"], 0.5)
        self.assertAlmostEqual(results["percentage_of_cis_gender_definition_words"], 0.5)
        self.assertAlmostEqual(results["additional_metrics"]["avg_non_binary_bias_ratio"], 0.0)
        self.assertAlmostEqual(results["additional_metrics"]["avg_trans_cis_bias_ratio"], 0.0)
        self.assertAlmostEqual(results["additional_metrics"]["avg_trans_cis_bias_conditional"], 0.0)
        self.assertAlmostEqual(results["additional_metrics"]["avg_trans_cis_bias_ratio_absolute"], 0.0)
        self.assertAlmostEqual(results["additional_metrics"]["avg_trans_cis_bias_conditional_absolute"], 0.0)
