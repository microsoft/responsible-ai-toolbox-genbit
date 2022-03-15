# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gender statistics class that can be leveraged to store corpus statistics
"""
from dataclasses import dataclass, asdict

@dataclass
class GenderStatistics:
    """Gender statistics  dataclass"""

    frequency_cutoff: float = 0.0
    num_words_considered: int = 0
    freq_of_female_gender_definition_words: int = 0
    freq_of_male_gender_definition_words: int = 0
    freq_of_non_binary_gender_definition_words: int = 0
    jsd: float = 0.0

    def get_return_dict(self, non_binary_gender_stats=True):
        return {key:value for key, value in asdict(self).items()
                if non_binary_gender_stats or not key.__contains__("non_binary")}
