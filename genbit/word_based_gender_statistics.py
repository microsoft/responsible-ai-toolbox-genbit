# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Word Based Gender statistics class that can be leveraged to store corpus statistics
"""
from dataclasses import dataclass, asdict

@dataclass
class WordBasedGenderStatistics:
    """Word Based Gender statistics dataclass"""

    frequency: float = 0.0
    female_count: float = 0.0
    male_count: float = 0.0
    non_binary_count: float = 0.0
    bias_ratio: float = 0.0
    bias_conditional_ratio: float = 0.0
    non_binary_bias_ratio: float = 0.0
    non_binary_bias_conditional_ratio: float = 0.0
    female_conditional_prob: float = 0.0
    male_conditional_prob: float = 0.0
    binary_conditional_prob: float = 0.0
    non_binary_conditional_prob: float = 0.0

    def get_return_dict(self, non_binary_gender_stats=True):
        return {key:value for key, value in asdict(self).items()
                if non_binary_gender_stats or not key.__contains__("non_binary")}
