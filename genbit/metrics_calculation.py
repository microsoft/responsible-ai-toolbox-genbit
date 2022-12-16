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
                 percentile_cutoff, tokenizer, non_binary_gender_stats=True,
                 trans_cis_stats=True):
        if language_code not in self._supported_languages:
            raise ValueError(
                self._UNSUPPORTED_LANGUAGE_ERROR.format(
                    language_code,
                    self._supported_languages))
        self._lemmatizer = Lemmatizer(language_code)
        self._context_window = context_window
        self._distance_weight = distance_weight
        self._percentile_cutoff = percentile_cutoff
        self._tokenizer = tokenizer
        self._stopwords = stopwords.stopwords(language_code.lower())
        self._female_gendered_words = set()
        self._male_gendered_words = set()
        self._non_binary_gendered_words = set()
        self._trans_gendered_words = set()
        self._cis_gendered_words = set()
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
        self._trans_conditional_probs = None
        self._cis_conditional_probs = None
        self._trans_cis_bias_scores_ratio = None
        self._trans_cis_bias_scores_conditional = None
        self._trans_cis_stats = False
        self._tokens_considered = None
        self._load_word_lists(language_code, non_binary_gender_stats, trans_cis_stats)
        self._multiword_expressions = self._initialize_multiword_expressions()

    def _load_single_word_list(self, language_code, filename, use_file=True):
        path = os.path.join(self._gendered_word_list_location, language_code, filename)
        if use_file and os.path.isfile(path):
            with open(path, 'r', encoding="utf-8") as word_lists:
                return True, set(self._form_mwe(word) for word in word_lists.readlines())
        return False, set()

    def _load_word_lists(self, language_code: str, non_binary_gender_stats: bool,
                         trans_cis_stats: bool):
        _, self._female_gendered_words = self._load_single_word_list(language_code, "female.txt")
        _, self._male_gendered_words   = self._load_single_word_list(language_code, "male.txt")
        # non-binary word lists only exists for English
        self._non_binary_gender_stats, self._non_binary_gendered_words = \
            self._load_single_word_list(language_code, "non-binary.txt", non_binary_gender_stats)
        # cis/trans only exists for English
        trans_stats, trans_words = \
            self._load_single_word_list(language_code, "trans.txt", trans_cis_stats)
        cis_stats, cis_words = \
            self._load_single_word_list(language_code, "cis.txt", trans_cis_stats and trans_stats)
        if cis_stats:
            self._trans_cis_stats = True
            self._trans_gendered_words = trans_words
            self._cis_gendered_words = cis_words

    def _form_mwe(self, word):
        word = word.strip()
        tokens = self._tokenizer.tokenize_data([word])
        if len(tokens) == 0:
            return word
        return '@@@'.join( [t for tl in tokens for t in tl] )

    @staticmethod
    def _add_to_trie(mwes, mwe):
        words = mwe.split('@@@')
        trie = mwes
        for i, word in enumerate(words):
            if word not in trie:
                trie[word] = dict()
            trie = trie[word]
            if i == len(words) - 1:
                trie[None] = mwe

    @staticmethod
    def _lookup_trie(trie, tok, start_index):
        longest = None, start_index
        for j in range(start_index, len(tok)):
            if tok[j] not in trie:
                return longest
            # otherwise tok[j] is in the trie and we can continue
            trie = trie[tok[j]]
            if None in trie:
                longest = trie[None], j+1
        return longest

    def _initialize_multiword_expressions(self):
        mwes = dict()  # trie of MWEs
        all_words = self._female_gendered_words.union( self._male_gendered_words )
        if self._non_binary_gender_stats:
            all_words = all_words.union( self._non_binary_gendered_words )
        if self._trans_cis_stats:
            all_words = all_words.union( self._trans_gendered_words )
            all_words = all_words.union( self._cis_gendered_words )
        for word in all_words:
            if '@@@' in word: # this is an mwe
                self._add_to_trie(mwes, word)
        return mwes

    def _join_multiword_expressions(self, lowered_tokens : List[str]):
        mwe_tokens = []
        i = 0
        while i < len(lowered_tokens):
            mwe, j = self._lookup_trie(self._multiword_expressions, lowered_tokens, i)
            if mwe is None:
                mwe_tokens.append( lowered_tokens[i] )
                i += 1
            else:
                mwe_tokens.append( mwe )
                assert j > i
                i = j
        return mwe_tokens

    def analyze_text(self, tokenized_text: List[str]):
        lowered_tokens = [token.lower() for token in tokenized_text]
        mwe_tokens = self._join_multiword_expressions(lowered_tokens)
        tokenized_text = [token for token in mwe_tokens if (
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
                    "count": 1, "female": 0, "male": 0, "non-binary": 0, "trans":0, "cis": 0}

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
       
            if token in self._trans_gendered_words:
                self._update_all_surrounding_words(
                    token_index, "trans", sentence_tokens)
            elif token in self._cis_gendered_words:
                self._update_all_surrounding_words(
                    token_index, "cis", sentence_tokens)

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
               token not in self._non_binary_gendered_words and \
               token not in self._trans_gendered_words and \
               token not in self._cis_gendered_words:
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
                    metrics["trans"] = metrics.get("trans", 0) + 1
                    metrics["cis"] = metrics.get("cis", 0) + 1
                    self._cooccurrence_matrix[lemmatized_token] = metrics
        self._unlemmatized_cooccurrence_matrix = dict()

    def _get_cooccurrence_count(self, values):
        count = values["female"] + values["male"]

        if self._non_binary_gender_stats:
            count += values["non-binary"]

        if self._trans_cis_stats:
            count += values["trans"] + values["cis"]

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
        total_trans_freq_count = 0
        total_cis_freq_count = 0
        for token, counts in self._cooccurrence_matrix.items():
            if token in self._female_gendered_words:
                total_female_freq_count += counts["count"]
            if token in self._male_gendered_words:
                total_male_freq_count += counts["count"]
            if token in self._non_binary_gendered_words:
                total_non_binary_freq_count += counts["count"]
                if token not in self._trans_gendered_words: # avoid double counting
                    total_trans_freq_count += counts["count"]
            if token in self._trans_gendered_words:
                total_trans_freq_count += counts["count"]
            if token in self._cis_gendered_words:
                total_cis_freq_count += counts["count"]
        results = {
            "female": total_female_freq_count,
            "male": total_male_freq_count,
            "non-binary": total_non_binary_freq_count,
            "trans": total_trans_freq_count,
            "cis": total_cis_freq_count,
            }
        return results

    def _calculate_metrics(self):
        overall_metrics = OverallGenderStatistics()
        self._bias_scores_ratio = {}
        self._bias_scores_conditional = {}
        self._non_binary_bias_scores_ratio = {}
        self._non_binary_bias_scores_conditional = {}
        self._trans_cis_bias_scores_ratio = {}
        self._trans_cis_bias_scores_conditional = {}
        self._female_conditional_probs = {}
        self._male_conditional_probs = {}
        self._non_binary_conditional_probs = {}
        self._trans_conditional_probs = {}
        self._cis_conditional_probs = {}
        self._tokens_considered = 0
        gender_lists = self._female_gendered_words.union(self._male_gendered_words, \
                self._non_binary_gendered_words, self._trans_gendered_words, self._cis_gendered_words)

        for token, metrics in self._cooccurrence_matrix.items():
            # only calculate metrics over 'gender neutral' tokens and for
            # tokens that have co occurrence counts above the configured cutoff            
            if (token not in gender_lists and (self._get_cooccurrence_count(metrics) >= self._cutoff)):
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
                    # metrics["male"] + metrics["female"]  should be >= 2 since they're both smoothed by 1
                    self._non_binary_bias_scores_ratio[token] = math.log(
                        (metrics["male"] + metrics["female"] - 1) / metrics["non-binary"])
                self._non_binary_conditional_probs[token] = metrics["non-binary"] / (
                    self._gendered_word_counts["non-binary"] + len(self._cooccurrence_matrix))
                if self._non_binary_conditional_probs[token] > 0:
                    binary_conditional_token = (metrics["female"] + metrics["male"] - 1) \
                        / (self._gendered_word_counts["female"] + self._gendered_word_counts["male"] + \
                            len(self._cooccurrence_matrix))
                    self._non_binary_bias_scores_conditional[token] = \
                        math.log(binary_conditional_token / self._non_binary_conditional_probs[token])

                # compute trans/cis gender metrics for the token
                self._trans_cis_bias_scores_ratio[token] = 0
                self._trans_cis_bias_scores_conditional[token] = 0
                if metrics["non-binary"] + metrics["trans"] > 1:
                    self._trans_cis_bias_scores_ratio[token] = math.log(
                        metrics["cis"] /
                          (metrics["non-binary"] + metrics["trans"] - 1))

                self._trans_conditional_probs[token] = \
                    (metrics["non-binary"] + metrics["trans"] - 1) / \
                    (self._gendered_word_counts["non-binary"] + \
                        self._gendered_word_counts["trans"] + \
                        len(self._cooccurrence_matrix))
                self._cis_conditional_probs[token] = (metrics["cis"]) / \
                        (self._gendered_word_counts["cis"] + \
                         len(self._cooccurrence_matrix))
                if self._trans_conditional_probs[token] > 0:
                    self._trans_cis_bias_scores_conditional[token] = math.log(
                        self._cis_conditional_probs[token] / self._trans_conditional_probs[token])

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

                overall_metrics.avg_trans_cis_bias_ratio += self._trans_cis_bias_scores_ratio[token]
                overall_metrics.avg_trans_cis_bias_ratio_absolute += abs(
                    self._trans_cis_bias_scores_ratio[token])
                overall_metrics.avg_trans_cis_bias_conditional += self._trans_cis_bias_scores_conditional[token]
                overall_metrics.avg_trans_cis_bias_conditional_absolute += \
                    abs(self._trans_cis_bias_scores_conditional[token])

        number_of_scores = max(1, len(self._bias_scores_ratio)) # avoid devision by 0

        overall_metrics.avg_bias_ratio /= number_of_scores
        overall_metrics.avg_bias_ratio_absolute /= number_of_scores
        overall_metrics.avg_bias_conditional /= number_of_scores
        overall_metrics.avg_bias_conditional_absolute /= number_of_scores
        overall_metrics.avg_non_binary_bias_ratio /= number_of_scores
        overall_metrics.avg_non_binary_bias_ratio_absolute /= number_of_scores
        overall_metrics.avg_non_binary_bias_conditional /= number_of_scores
        overall_metrics.avg_non_binary_bias_conditional_absolute /= number_of_scores
        overall_metrics.avg_trans_cis_bias_ratio /= number_of_scores
        overall_metrics.avg_trans_cis_bias_ratio_absolute /= number_of_scores
        overall_metrics.avg_trans_cis_bias_conditional /= number_of_scores
        overall_metrics.avg_trans_cis_bias_conditional_absolute /= number_of_scores

        if number_of_scores >= 2:
            overall_metrics.std_dev_bias_ratio = statistics.stdev(
                self._bias_scores_ratio.values())
            overall_metrics.std_dev_bias_conditional = statistics.stdev(
                self._bias_scores_conditional.values())
            overall_metrics.std_dev_non_binary_bias_ratio = statistics.stdev(
                self._non_binary_bias_scores_ratio.values())
            overall_metrics.std_dev_non_binary_bias_conditional = statistics.stdev(
                self._non_binary_bias_scores_conditional.values())
            overall_metrics.std_dev_trans_cis_bias_ratio = statistics.stdev(
                self._trans_cis_bias_scores_ratio.values())
            overall_metrics.std_dev_trans_cis_bias_conditional = statistics.stdev(
                self._trans_cis_bias_scores_conditional.values())
            overall_metrics.std_dev_bias_ratio = 0
            overall_metrics.std_dev_bias_conditional = 0
            overall_metrics.std_dev_non_binary_bias_ratio = 0
            overall_metrics.std_dev_non_binary_bias_conditional = 0
            overall_metrics.std_dev_trans_cis_bias_ratio = 0
            overall_metrics.std_dev_trans_cis_bias_conditional = 0

        total_gender_definition_words = \
            max(1, self._gendered_word_counts["female"] + \
                   self._gendered_word_counts["male"] + \
                   self._gendered_word_counts["non-binary"])  # avoid devision by 0
    
        overall_metrics.percentage_of_female_gender_definition_words = \
            self._gendered_word_counts["female"] / total_gender_definition_words
        overall_metrics.percentage_of_male_gender_definition_words = \
            self._gendered_word_counts["male"] / total_gender_definition_words
        overall_metrics.percentage_of_non_binary_gender_definition_words = \
            self._gendered_word_counts["non-binary"] / total_gender_definition_words


        total_trans_cis_definition_words = \
            max(1, self._gendered_word_counts["trans"] + \
                   self._gendered_word_counts["cis"] + \
                   self._gendered_word_counts["non-binary"])  # avoid devision by 0

        overall_metrics.percentage_of_trans_gender_definition_words = \
             (self._gendered_word_counts["trans"] + self._gendered_word_counts["non-binary"]) \
                 / total_trans_cis_definition_words
        overall_metrics.percentage_of_cis_gender_definition_words = \
             self._gendered_word_counts["cis"] / total_trans_cis_definition_words

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
        metric_statistics.freq_of_cis_gender_definition_words = self._gendered_word_counts[
            "cis"]
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
        gender_lists = self._female_gendered_words.union(self._male_gendered_words, \
                self._non_binary_gendered_words, self._trans_gendered_words, self._cis_gendered_words)            
        for token, values in self._cooccurrence_matrix.items():            
            if (token not in gender_lists and (self._get_cooccurrence_count(values) >= self._cutoff)):
                word_based_gender_statistics = WordBasedGenderStatistics()
                word_based_gender_statistics.frequency = values["count"]
                word_based_gender_statistics.female_count = values["female"]
                word_based_gender_statistics.male_count = values["male"]
                word_based_gender_statistics.non_binary_count = values["non-binary"]
                word_based_gender_statistics.trans_count = values["trans"]
                word_based_gender_statistics.cis_count = values["cis"]
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
