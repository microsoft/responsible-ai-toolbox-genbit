# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The main calculation class for the gender bias measure.
The co-occurrence matrix is updated to reflect the corpus that is evaluated.
As a second part of the analysis, all measures are calculated based on this co-occurrence matrix.
"""

import math
import os
from typing import List
import statistics
import numpy as np
from scipy.spatial import distance
import stopwordsiso as stopwords
from genbit.gender_statistics import GenderStatistics
from genbit.overall_gender_statistics import OverallGenderStatistics
from genbit.word_based_gender_statistics import WordBasedGenderStatistics
from genbit.lemmatizer import Lemmatizer


class MetricsCalculation:
    """Main metric calculation for gender bias measure, updates co-occurrence matrix by analyze_text,
    then calculates the metrics using get_statistics"""

    _gendered_word_list_location = os.path.join(
        os.path.dirname(__file__), 'gendered-word-lists')
    _supported_languages = [folder for folder in os.listdir(_gendered_word_list_location) if len(folder) == 2]
    _UNSUPPORTED_LANGUAGE_ERROR = "The languages {0} is not supported, currently supported languages are:{1}"

    def __init__(self, language_code, context_window, distance_weight,
                 percentile_cutoff, non_binary_gender_stats=True):
        if language_code not in self._supported_languages:
            raise ValueError(
                self._UNSUPPORTED_LANGUAGE_ERROR.format(
                    language_code,
                    self._supported_languages))
        self._lemmatizer = Lemmatizer(language_code)
        self._context_window = context_window
        self._distance_weight = distance_weight
        self._percentile_cutoff = percentile_cutoff
        self._stopwords = stopwords.stopwords(language_code.lower())
        self._female_gendered_words = set()
        self._male_gendered_words = set()
        self._non_binary_gendered_words = set()
        self._unlemmatized_cooccurrence_matrix = dict()
        self._cooccurrence_matrix = dict()
        self._cutoff = 0
        self._gendered_word_counts = None
        self._bias_scores_ratio = None
        self._bias_scores_conditional = None
        self._non_binary_gender_stats = False
        self._non_binary_bias_scores_ratio = None
        self._non_binary_bias_scores_conditional = None
        self._female_conditional_probs = None
        self._male_conditional_probs = None
        self._non_binary_conditional_probs = None
        self._tokens_considered = None
        self._load_word_lists(language_code, non_binary_gender_stats)

    def _load_word_lists(self, language_code: str, non_binary_gender_stats: bool):
        with open(os.path.join(self._gendered_word_list_location, language_code, "female.txt"), 'r',
                encoding="utf-8") as female_word_lists:
            self._female_gendered_words = set(
                word.strip() for word in female_word_lists.readlines())
        with open(os.path.join(self._gendered_word_list_location, language_code, "male.txt"), 'r',
                encoding="utf-8") as male_word_lists:
            self._male_gendered_words = set(
                word.strip() for word in male_word_lists.readlines())
        # non-binary word lists only exists for English
        non_binary_word_lists_path = os.path.join(
            self._gendered_word_list_location, language_code, "non-binary.txt")
        if os.path.isfile(non_binary_word_lists_path) and non_binary_gender_stats:
            with open(non_binary_word_lists_path, 'r',
                    encoding="utf-8") as non_binary_word_lists:
                self._non_binary_gendered_words = set(
                    word.strip() for word in non_binary_word_lists.readlines())
                self._non_binary_gender_stats = True

    def analyze_text(self, tokenized_text: List[str]):
        lowered_tokens = [token.lower() for token in tokenized_text]
        tokenized_text = [token for token in lowered_tokens if (
            token in self._male_gendered_words) or (
            token in self._female_gendered_words) or (
            token in self._non_binary_gendered_words) or not (
            token in self._stopwords)]
        self._analyze_independent_text(tokenized_text)

    def _analyze_independent_text(self, sentence_tokens: List[str]):
        self._count_tokens(sentence_tokens)
        self._calculate_cooccurrences(sentence_tokens)

    def _count_tokens(self, sentence_tokens):
        for token in sentence_tokens:
            # update frequency count for token
            if token in self._unlemmatized_cooccurrence_matrix:
                self._unlemmatized_cooccurrence_matrix[token]["count"] += 1
            else:
                self._unlemmatized_cooccurrence_matrix[token] = {
                    "count": 1, "female": 0, "male": 0, "non-binary": 0}

    def _calculate_cooccurrences(self, sentence_tokens):
        for token_index, token in enumerate(sentence_tokens):
            if token in self._female_gendered_words:
                self._update_all_surrounding_words(
                    token_index, "female", sentence_tokens)
            elif token in self._male_gendered_words:
                self._update_all_surrounding_words(
                    token_index, "male", sentence_tokens)
            elif token in self._non_binary_gendered_words:
                self._update_all_surrounding_words(
                    token_index, "non-binary", sentence_tokens)

    def _update_all_surrounding_words(
            self, gendered_token_index, gender, tokens):
        window_start_index = max(
            0, gendered_token_index - self._context_window)
        window_end_index = min(
            len(tokens) - 1,
            gendered_token_index + self._context_window)

        for window_index in range(window_start_index, window_end_index + 1):
            if window_index == gendered_token_index:
                continue
            token_distance = abs(gendered_token_index - window_index)
            self._unlemmatized_cooccurrence_matrix[tokens[window_index]
                                      ][gender] += pow(self._distance_weight, token_distance)

    def get_statistics(self, output_statistics=True, output_word_list=True):
        self._lemmatize_cooccurrence_matrix()
        self._cutoff = self._calculate_cutoff()
        self._gendered_word_counts = self._calculate_gendered_word_counts()
        gender_bias_results = self._calculate_metrics()
        if output_statistics:
            gender_bias_results["statistics"] = self._calculate_statistics()
        if output_word_list:
            gender_bias_results["token_based_metrics"] = self._get_word_based_metrics()
        return gender_bias_results

    def _lemmatize_cooccurrence_matrix(self):
        for token, metrics in self._unlemmatized_cooccurrence_matrix.items():
            lemmatized_token = token
            if token not in self._female_gendered_words and \
               token not in self._male_gendered_words and \
               token not in self._non_binary_gendered_words:
                lemmatized_token = self._lemmatizer.lemmatize_token(token)
            if lemmatized_token:
                if lemmatized_token in self._cooccurrence_matrix :
                    for metric, value in metrics.items():
                        self._cooccurrence_matrix[lemmatized_token][metric] += value
                else:
                    # add smoothing when token is added to the lemmatized matrix for the first time
                    metrics["female"] = metrics.get("female", 0) + 1
                    metrics["male"] = metrics.get("male", 0) + 1
                    metrics["non-binary"] = metrics.get("non-binary", 0) + 1
                    self._cooccurrence_matrix[lemmatized_token] = metrics
        self._unlemmatized_cooccurrence_matrix = dict()

    def _get_cooccurrence_count(self, values):
        count = values["female"] + values["male"]

        if self._non_binary_gender_stats:
            count += values["non-binary"]

        return count

    def _calculate_cutoff(self):
        counts = [self._get_cooccurrence_count(values)
                  for token, values in self._cooccurrence_matrix.items()]
        trimmed_counts = [
            cooccurrence for cooccurrence in counts if cooccurrence > 2]
        if trimmed_counts == []:
            return 2
        return np.percentile(trimmed_counts, self._percentile_cutoff)

    def _calculate_gendered_word_counts(self):
        total_male_freq_count = 0
        total_female_freq_count = 0
        total_non_binary_freq_count = 0
        for token, counts in self._cooccurrence_matrix.items():
            if token in self._female_gendered_words:
                total_female_freq_count += counts["count"]
            if token in self._male_gendered_words:
                total_male_freq_count += counts["count"]
            if token in self._non_binary_gendered_words:
                total_non_binary_freq_count += counts["count"]
        results = {
            "female": total_female_freq_count,
            "male": total_male_freq_count,
            "non-binary": total_non_binary_freq_count}
        return results

    def _calculate_metrics(self):
        overall_metrics = OverallGenderStatistics()
        self._bias_scores_ratio = {}
        self._bias_scores_conditional = {}
        self._non_binary_bias_scores_ratio = {}
        self._non_binary_bias_scores_conditional = {}
        self._female_conditional_probs = {}
        self._male_conditional_probs = {}
        self._non_binary_conditional_probs = {}
        self._tokens_considered = 0

        for token, metrics in self._cooccurrence_matrix.items():
            # only calculate metrics over 'gender neutral' tokens and for
            # tokens that have co occurrence counts above the configured cutoff
            if (token not in self._female_gendered_words) and (
                token not in self._male_gendered_words) and (
                token not in self._non_binary_gendered_words) and (
                self._get_cooccurrence_count(metrics) >= self._cutoff):
                self._tokens_considered += 1
                self._bias_scores_ratio[token] = math.log(
                    metrics["male"] / metrics["female"])
                self._female_conditional_probs[token] = metrics["female"] / (
                    self._gendered_word_counts["female"] + len(self._cooccurrence_matrix))
                self._male_conditional_probs[token] = metrics["male"] / (
                    self._gendered_word_counts["male"] + len(self._cooccurrence_matrix))
                self._bias_scores_conditional[token] = math.log(
                    self._male_conditional_probs[token] / self._female_conditional_probs[token])

                # compute non-binary gender metrics for the token
                self._non_binary_bias_scores_ratio[token] = 0
                self._non_binary_bias_scores_conditional[token] = 0
                if metrics["non-binary"] > 0:
                    self._non_binary_bias_scores_ratio[token] = math.log(
                        (metrics["male"] + metrics["female"]) / metrics["non-binary"])
                self._non_binary_conditional_probs[token] = metrics["non-binary"] / (
                    self._gendered_word_counts["non-binary"] + len(self._cooccurrence_matrix))
                if self._non_binary_conditional_probs[token] > 0:
                    self._non_binary_bias_scores_conditional[token] = math.log(
                        (self._female_conditional_probs[token] + self._male_conditional_probs[token]) \
                            / self._non_binary_conditional_probs[token])

                # compute totals and the average is computed at the end of the loop
                overall_metrics.avg_bias_ratio += self._bias_scores_ratio[token]
                overall_metrics.avg_bias_ratio_absolute += abs(self._bias_scores_ratio[token])
                overall_metrics.avg_bias_conditional += self._bias_scores_conditional[token]
                overall_metrics.avg_bias_conditional_absolute += abs(
                    self._bias_scores_conditional[token])
                overall_metrics.avg_non_binary_bias_ratio += self._non_binary_bias_scores_ratio[token]
                overall_metrics.avg_non_binary_bias_ratio_absolute += abs(
                    self._non_binary_bias_scores_ratio[token])
                overall_metrics.avg_non_binary_bias_conditional += self._non_binary_bias_scores_conditional[token]
                overall_metrics.avg_non_binary_bias_conditional_absolute += \
                    abs(self._non_binary_bias_scores_conditional[token])

        number_of_scores = max(1, len(self._bias_scores_ratio)) # avoid devision by 0

        overall_metrics.avg_bias_ratio /= number_of_scores
        overall_metrics.avg_bias_ratio_absolute /= number_of_scores
        overall_metrics.avg_bias_conditional /= number_of_scores
        overall_metrics.avg_bias_conditional_absolute /= number_of_scores
        overall_metrics.avg_non_binary_bias_ratio /= number_of_scores
        overall_metrics.avg_non_binary_bias_ratio_absolute /= number_of_scores
        overall_metrics.avg_non_binary_bias_conditional /= number_of_scores
        overall_metrics.avg_non_binary_bias_conditional_absolute /= number_of_scores

        if number_of_scores >= 2:
            overall_metrics.std_dev_bias_ratio = statistics.stdev(
                self._bias_scores_ratio.values())
            overall_metrics.std_dev_bias_conditional = statistics.stdev(
                self._bias_scores_conditional.values())
            overall_metrics.std_dev_non_binary_bias_ratio = statistics.stdev(
                self._non_binary_bias_scores_ratio.values())
            overall_metrics.std_dev_non_binary_bias_conditional = statistics.stdev(
                self._non_binary_bias_scores_conditional.values())
        else:
            overall_metrics.std_dev_bias_ratio = 0
            overall_metrics.std_dev_bias_conditional = 0
            overall_metrics.std_dev_non_binary_bias_ratio = 0
            overall_metrics.std_dev_non_binary_bias_conditional = 0

        total_gender_definition_words = \
            max(1, self._gendered_word_counts["female"] + \
                   self._gendered_word_counts["male"] + \
                   self._gendered_word_counts["non-binary"]) # avoid devision by 0

        overall_metrics.percentage_of_female_gender_definition_words = \
            self._gendered_word_counts["female"] / total_gender_definition_words
        overall_metrics.percentage_of_male_gender_definition_words = \
            self._gendered_word_counts["male"] / total_gender_definition_words
        overall_metrics.percentage_of_non_binary_gender_definition_words = \
            self._gendered_word_counts["non-binary"] / total_gender_definition_words

        return overall_metrics.get_return_dict(self._non_binary_gender_stats)

    def _calculate_statistics(self):
        metric_statistics = GenderStatistics()
        metric_statistics.frequency_cutoff = self._cutoff
        metric_statistics.num_words_considered = self._tokens_considered
        metric_statistics.freq_of_female_gender_definition_words = self._gendered_word_counts[
            "female"]
        metric_statistics.freq_of_male_gender_definition_words = self._gendered_word_counts[
            "male"]
        metric_statistics.freq_of_non_binary_gender_definition_words = self._gendered_word_counts[
            "non-binary"]
        metric_statistics.jsd = self._calculate_gender_distribution_divergence()
        return metric_statistics.get_return_dict(self._non_binary_gender_stats)

    def _calculate_gender_distribution_divergence(self):
        female_count = sum([values["female"] for token,
                            values in self._cooccurrence_matrix.items()])
        male_count = sum([values["male"] for token,
                          values in self._cooccurrence_matrix.items()])
        p_w_given_female = np.array(
            [values["female"] / female_count for token, values in self._cooccurrence_matrix.items()])
        p_w_given_male = np.array(
            [values["male"] / male_count for token, values in self._cooccurrence_matrix.items()])

        jsd = pow(distance.jensenshannon(p_w_given_female, p_w_given_male), 2)
        return jsd

    def _get_word_based_metrics(self):
        word_based_statistics = {}
        for token, values in self._cooccurrence_matrix.items():
            if (token not in self._female_gendered_words) and (
                token not in self._male_gendered_words) and (
                token not in self._non_binary_gendered_words) and (
                self._get_cooccurrence_count(values) >= self._cutoff):
                word_based_gender_statistics = WordBasedGenderStatistics()
                word_based_gender_statistics.frequency = values["count"]
                word_based_gender_statistics.female_count = values["female"]
                word_based_gender_statistics.male_count = values["male"]
                word_based_gender_statistics.non_binary_count = values["non-binary"]
                word_based_gender_statistics.bias_ratio = self._bias_scores_ratio[token]
                word_based_gender_statistics.bias_conditional_ratio = self._bias_scores_conditional[token]
                word_based_gender_statistics.non_binary_bias_ratio = self._non_binary_bias_scores_ratio[token]
                word_based_gender_statistics.non_binary_bias_conditional_ratio = \
                    self._non_binary_bias_scores_conditional[token]
                word_based_gender_statistics.female_conditional_prob = self._female_conditional_probs[token]
                word_based_gender_statistics.male_conditional_prob = self._male_conditional_probs[token]
                word_based_gender_statistics.binary_conditional_prob = \
                    self._female_conditional_probs[token] + self._male_conditional_probs[token]
                word_based_gender_statistics.non_binary_conditional_prob = self._non_binary_conditional_probs[token]
                word_based_statistics[token] = \
                    word_based_gender_statistics.get_return_dict(self._non_binary_gender_stats)

        return word_based_statistics
