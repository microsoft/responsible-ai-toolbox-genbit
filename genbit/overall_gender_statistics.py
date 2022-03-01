# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Overall Gender statistics class that can be leveraged to store corpus statistics
"""
from dataclasses import dataclass, asdict

@dataclass
class OverallGenderStatistics:
    """Overall Gender statistics dataclass"""

    avg_bias_ratio: float = 0.0
    avg_bias_conditional: float = 0.0
    avg_bias_ratio_absolute: float = 0.0
    avg_bias_conditional_absolute: float = 0.0
    avg_non_binary_bias_ratio: float = 0.0
    avg_non_binary_bias_conditional: float = 0.0
    avg_non_binary_bias_ratio_absolute: float = 0.0
    avg_non_binary_bias_conditional_absolute: float = 0.0
    std_dev_bias_ratio: float = 0.0
    std_dev_bias_conditional: float = 0.0
    std_dev_non_binary_bias_ratio: float = 0.0
    std_dev_non_binary_bias_conditional: float = 0.0
    percentage_of_female_gender_definition_words: float = 0.0
    percentage_of_male_gender_definition_words: float = 0.0
    percentage_of_non_binary_gender_definition_words: float = 0.0

    def get_return_dict(self, non_binary_gender_stats=True):
        overall_metrics = {}
        overall_metrics["genbit_score"] = self.avg_bias_conditional_absolute
        overall_metrics["percentage_of_female_gender_definition_words"] = \
            self.percentage_of_female_gender_definition_words
        overall_metrics["percentage_of_male_gender_definition_words"] = \
            self.percentage_of_male_gender_definition_words
        overall_metrics["percentage_of_non_binary_gender_definition_words"] = \
            self.percentage_of_non_binary_gender_definition_words
        additional_metrics = {key:value for key, value in asdict(self).items()
                              if key not in overall_metrics and (
                                  non_binary_gender_stats or not key.__contains__("non_binary"))}
        overall_metrics["additional_metrics"] = additional_metrics
        return overall_metrics
