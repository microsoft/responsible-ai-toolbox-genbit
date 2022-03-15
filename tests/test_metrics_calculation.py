# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import os
import json
from parameterized import parameterized
from genbit.metrics_calculation import MetricsCalculation


class AnalyzeSentencesTestCase(unittest.TestCase):

    input_sentences_path_tokenized = "tests/test_data_files/input/winogender_en_50_tokenized.tsv"
    input_sentences_path_cooccurrence_matrix = "tests/test_data_files/input/winogender_en_50_cooccurrence_matrix.json"
    expected_cooccurrence_matrix_path = "tests/test_data_files/expected/winogender_en_50_cooccurrence.json"
    expected_overall_results_path = "tests/test_data_files/expected/winogender_en_50_overall_metrics.json"
    expected_results_path = "tests/test_data_files/expected/winogender_en_50_overall_metrics"
    expected_statistics_results_path = "tests/test_data_files/expected/winogender_en_50_statistics_metrics.json"
    expected_word_based_results_path = "tests/test_data_files/expected/winogender_en_50_word_based_metrics.json"
    language_code = "en"
    context_window = 5
    distance_weight = 0.95
    percentile_cutoff = 80

    def setUp(self):
        self.metrics_calculation = MetricsCalculation(
            self.language_code,
            self.context_window,
            self.distance_weight,
            self.percentile_cutoff,
            False)

    def testInit(self):
        self.assertIsInstance(self.metrics_calculation, MetricsCalculation)
        self.assertTrue(
            hasattr(
                self.metrics_calculation,
                "_female_gendered_words"))
        self.assertTrue(
            hasattr(
                self.metrics_calculation,
                "_male_gendered_words"))
        self.assertTrue(
            hasattr(
                self.metrics_calculation,
                "_cooccurrence_matrix"))

    def testUnsupportedLangueage(self):
        unsupported_language_code = "nl"
        with self.assertRaises(ValueError):
            MetricsCalculation(
                unsupported_language_code,
                self.context_window,
                self.distance_weight,
                self.percentile_cutoff)

    def testAnalyzeText(self):
        input_file = os.path.join(
            os.getcwd(), self.input_sentences_path_tokenized)
        with open(input_file, "r", encoding="utf-8") as input_sentence_file:
            tokenized_input_text = json.load(input_sentence_file)

        self.metrics_calculation.analyze_text(tokenized_input_text)

        expected_matrix_path = os.path.join(
            os.getcwd(),
            self.expected_cooccurrence_matrix_path)

        with open(expected_matrix_path, "r", encoding="utf-8") as expected_cooccurrence_matrix_file:
            expected_cooccurrence_matrix = json.load(
                expected_cooccurrence_matrix_file)

        for token, values in self.metrics_calculation._unlemmatized_cooccurrence_matrix.items():
            self.assertAlmostEqual(
                expected_cooccurrence_matrix[token]["female"],
                values["female"])
            self.assertAlmostEqual(
                expected_cooccurrence_matrix[token]["male"],
                values["male"])
            self.assertAlmostEqual(
                expected_cooccurrence_matrix[token]["count"],
                values["count"])
        # Test both ways in case the co-occurrence matrix is missing items.

        for token, values in expected_cooccurrence_matrix.items():
            self.assertAlmostEqual(
                values["female"],
                self.metrics_calculation._unlemmatized_cooccurrence_matrix[token]["female"])
            self.assertAlmostEqual(
                values["male"],
                self.metrics_calculation._unlemmatized_cooccurrence_matrix[token]["male"])
            self.assertAlmostEqual(
                values["count"],
                self.metrics_calculation._unlemmatized_cooccurrence_matrix[token]["count"])

    @parameterized.expand(
        [
            [{"statistics": False}],
            [{"statistics": True}],
        ])
    def testCalculateMetrics(self, input_parameters):
        statistics = input_parameters["statistics"]
        input_file = os.path.join(
            os.getcwd(), self.input_sentences_path_cooccurrence_matrix)
        with open(input_file, "r", encoding="utf-8") as input_matrix_file:
            cooccurrence_matrix_input = json.load(input_matrix_file)

        self.metrics_calculation._unlemmatized_cooccurrence_matrix = cooccurrence_matrix_input
        overall_metrics = self.metrics_calculation.get_statistics(
            statistics, False)

        if not statistics:
            expected_results_path = os.path.join(
                os.getcwd(), self.expected_results_path + ".json")
        else:
            overall_metrics = overall_metrics["statistics"]
            expected_results_path = os.path.join(
                os.getcwd(), self.expected_statistics_results_path)

        with open(expected_results_path, "r", encoding="utf-8") as expected_overall_results_file:
            expected_overall_result = json.load(expected_overall_results_file)

        for metric, value in expected_overall_result.items():
            self.assertIn(metric, overall_metrics)
            self.assertAlmostEqual(value, overall_metrics[metric])

    def testCalculateWordBasedBiasStatisticsMetrics(self):
        input_file = os.path.join(
            os.getcwd(), self.input_sentences_path_cooccurrence_matrix)
        with open(input_file, "r", encoding="utf-8") as input_matrix_file:
            cooccurrence_matrix_input = json.load(input_matrix_file)

        self.metrics_calculation._unlemmatized_cooccurrence_matrix = cooccurrence_matrix_input
        overall_metrics = self.metrics_calculation.get_statistics(False, True)
        token_list = overall_metrics["token_based_metrics"]

        expected_results_path = os.path.join(
            os.getcwd(), self.expected_word_based_results_path)
        with open(expected_results_path, "r", encoding="utf-8") as expected_statistics_results_file:
            expected_statistics_result = json.load(
                expected_statistics_results_file)

        for token, metrics in expected_statistics_result.items():
            self.assertIn(token, token_list)

            for metric, value in metrics.items():
                self.assertIn(metric, token_list[token])
                self.assertAlmostEqual(value, token_list[token][metric])
