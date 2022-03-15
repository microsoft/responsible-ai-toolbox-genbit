# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import os
import stanza
import json
from genbit.lemmatizer import Lemmatizer
from genbit.metrics_calculation import MetricsCalculation
import time


class LemmatizerTestCaseTestCase(unittest.TestCase):

    language_code = "en"
 
    def testInit(self):
        lemmatizer = Lemmatizer(self.language_code)
        self.assertIsInstance(lemmatizer, Lemmatizer)
        self.assertTrue(hasattr(lemmatizer, "_language_code"))
        self.assertIsInstance(lemmatizer._lemmatizer, stanza.pipeline.core.Pipeline)

    def test_lemmatize_token(self):
        lem = Lemmatizer(self.language_code)
        self.assertEqual(lem.lemmatize_token("is"), "be")
        self.assertEqual(lem.lemmatize_token("was"), "be")
        self.assertEqual(lem.lemmatize_token("were"), "be")
        self.assertEqual(lem.lemmatize_token("are"), "be")
        self.assertEqual(lem.lemmatize_token("be"), "be")

    def test_cooccurrence_matrix(self):
         metrics_calculation = MetricsCalculation(self.language_code, 5, 0.95, 80, False)
         metrics_calculation.analyze_text(["he", "eats"])
         metrics_calculation.analyze_text(["he", "eating"])
         metrics_calculation.analyze_text(["he", "eat"])
         metrics_calculation.analyze_text(["he", "ate"])
         metrics_calculation.analyze_text(["he", "has", "eaten"])

         expected_matrix = {'he': {'count': 5, 'female': 0, 'male': 0}, 'eats': {'count': 1, 'female': 0, 'male': 0.95}, 'eating': {'count': 1, 'female': 0, 'male': 0.95}, 'eat': {'count': 1, 'female': 0, 'male': 0.95}, 'ate': {'count': 1, 'female': 0, 'male': 0.95}, 'eaten': {'count': 1, 'female': 0, 'male': 0.95}}
         for token, values in expected_matrix.items():
            self.assertAlmostEqual(
                values["female"],
                metrics_calculation._unlemmatized_cooccurrence_matrix[token]["female"])
            self.assertAlmostEqual(
                values["male"],
                metrics_calculation._unlemmatized_cooccurrence_matrix[token]["male"])

         metrics_calculation._lemmatize_cooccurrence_matrix()
         expected_lemmatized_matrix = {'he': {'count': 5, 'female': 1, 'male': 1}, 'eat': {'count': 5, 'female': 1, 'male': 5.75}} #{'he': {'count': 5, 'female': 0, 'male': 0}, 'eat': {'count': 5, 'female': 0, 'male': 4.75}}
         for token, values in expected_lemmatized_matrix.items():
            self.assertAlmostEqual(
                values["female"],
                metrics_calculation._cooccurrence_matrix[token]["female"])
            self.assertAlmostEqual(
                values["male"],
                metrics_calculation._cooccurrence_matrix[token]["male"])

