# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Lemmatization for the gender bias measure."""

import os
import stanza

class Lemmatizer:
    """Lemmatizer class, the lemmatizer is only initialized for the selected language when lemmatization is needed
    to avoid initializing models before invocation"""

    _stanza_lemmatizers_location = os.path.join(
        os.path.dirname(__file__), 'stanza_lemmatizers')
    _stanza_download_model_warning = "The stanza model for \"{0}\" is not present in {1}. This model will be "\
                    "downloaded, please abort this process if you do not wish to download the lemmatization model "\
                    "(this will restrict you from running the GenBiT)."

    def __init__(self, language_code:str):
        self._language_code = language_code
        self.initialize_lemmatizer()

    def lemmatize_token(self, token:str) -> str:
        lemmatized_stanza_info = self._lemmatizer(token)
        if len(lemmatized_stanza_info.sentences[0].words) > 0:
            lemmatized_token = lemmatized_stanza_info.sentences[0].words[0].lemma
        else:
            lemmatized_token = ""
        return lemmatized_token

    def initialize_lemmatizer(self):
        if not os.path.exists(self._stanza_lemmatizers_location):
            os.mkdir(self._stanza_lemmatizers_location)
        supported_languages = os.listdir(self._stanza_lemmatizers_location)
        if self._language_code not in supported_languages:
            stanza.download(
                lang=self._language_code,
                processors='tokenize,lemma',
                model_dir=self._stanza_lemmatizers_location)

        self._lemmatizer = stanza.Pipeline(lang=self._language_code,
                                           processors='tokenize,lemma',
                                           dir=self._stanza_lemmatizers_location,
                                            verbose=False)
