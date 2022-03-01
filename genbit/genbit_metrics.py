# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Main API for gender bias metrics, this API has two methods, add_data and get_metrics
that connect with a metrics_calculation object to perform the gender bias calculations
"""


from typing import List
from genbit.metrics_calculation import MetricsCalculation
from genbit.tokenizer import Tokenizer


class GenBitMetrics:
    """Main gender bias API, connecting the metrics calculation with the external users."""

    def __init__(self, language_code, context_window=30, distance_weight=0.95,
                 percentile_cutoff=80):
        if len(language_code) != 2:
            raise ValueError(
                "Invalid Language Code, the language code should be in ll format.")
        language_code = language_code.lower()
        self._tokenizer = Tokenizer(language_code)
        self._metrics_calculation = MetricsCalculation(
            language_code,
            context_window,
            distance_weight,
            percentile_cutoff)

    def add_data(self, text_data: List[str], tokenized=False):
        if isinstance(text_data, str):
            text_data = [text_data]
        if not tokenized:
            text_data = self._tokenizer.tokenize_data(text_data)
        for independent_tokenized_text_item in text_data:
            self._metrics_calculation.analyze_text(independent_tokenized_text_item)

    def get_metrics(self, output_statistics=True, output_word_list=True):
        results = self._metrics_calculation.get_statistics(
            output_statistics, output_word_list)
        return results
