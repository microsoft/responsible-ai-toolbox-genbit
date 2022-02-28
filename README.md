# GenBit: Gender Bias in Text Tool

The main goal of the Gender Bias Tool (GenBiT) is to analyze corpora and compute metrics that give insights into the gender bias present in a corpus. The computations in this tool are based primarily on ideas from Shikha Bordia and Samuel R. Bowman, "[Identifying and reducing gender bias in word-level language models](https://arxiv.org/abs/1904.03035)" in the NAACL 2019 Student Research Workshop.

## Installation

The code can be cloned or seen from here: [GenBiT source code](https://github.com/microsoft/responsibleaitoolbox-genbit)
The package is currently supported for local use on Windows and Linux and can be deployed as part of an azure function.
If your scenario is not listed here, please reach out to the Global AI team to discuss potential support for your current environment.

#### Authentication

If asked to provide your username and password when installing the pip package, please provide your ADO username and a [token genereated in ADO](https://office.visualstudio.com/_usersSettings/tokens) as password.

#### Tested and Supported environments

As mentioned before, the tested and supported environment for the GenBit python package are:

- Local usage on Windows
- Local usage on Linux (Tested on Ubuntu 18.04 and Debian Buster 10)
- As part of Azure functions installed using a [remote build](https://docs.microsoft.com/en-us/azure/azure-functions/functions-reference-python#remote-build-with-extra-index-url).

#### Installing the GenBiT python package with pip (Windows)

1. (optional) Create a virtual environment: `py -m venv environment-name`
2. Install the Azure artifacts keyring package: `pip install artifacts-keyring`
3. Install GenbiT (make sure you have the right authentication token from [here](https://office.visualstudio.com/_usersSettings/tokens) or validate your device): `pip install genbit --index-url https://office.pkgs.visualstudio.com/GSX/_packaging/GenderBias/pypi/simple/`
4. To verify whether this works open a python shell (`py`) and type `from genbit.genbit_metrics import GenBitMetrics`. For usage information, type: `help(genbit)`.

#### Installing the GenBiT python package with pip (Linux - Ubuntu 18.04)

1. Install [pip for python3](https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/)
2. Install virtualenv (this is necessary to run the installation in python3, use: `sudo apt-get install python3-venv`) and Create a virtual environment using: `virtualenv -p \usr\bin\python3 genbit-env`
3. Activate the virtual environment: `source genbit-env\bin\activate`.
4. Install the Azure artifacts keyring package: `pip install artifacts-keyring`
5. Install GenbiT (make sure you have the right authentication token from [here](https://office.visualstudio.com/_usersSettings/tokens) or validate your device): `pip install genbit --index-url https://office.pkgs.visualstudio.com/GSX/_packaging/GenderBias/pypi/simple/`
6. To verify whether this works open a python shell (`py`) and type `genbit.genbit_metrics import GenBitMetrics`. For usage information, type: `help(genbit)`.

#### Installing the GenBiT python package with pip (Linux - Debian Buster 10)

1. Install [pip for python3](https://linuxize.com/post/how-to-install-pip-on-debian-10/). (these commands are the same for Ubuntu and Debian)
2. Install the Azure artifacts keyring package: `pip3 install artifacts-keyring`
3. Install GenbiT (make sure you have the right authentication token from [here](https://office.visualstudio.com/_usersSettings/tokens) or validate your device): `pip3 install genbit --index-url https://office.pkgs.visualstudio.com/GSX/_packaging/GenderBias/pypi/simple/`
4. Verify the installation using ‘python3’ [enter] `genbit.genbit_metrics import GenBitMetrics`.

# Usage

To use GenBiT, it can be imported with:

`from genbit.genbit_metrics import GenBitMetrics`

Create a GenbiT object with the desired settings:

```
genbit_metrics_object = GenBitMetrics(language_code, context_window=5, distance_weight=0.95, percentile_cutoff=80)
```

Lets say we want to use GenbiT with a test sentence, we can add the sentence to GenbiT:

```
test_text = ["I think she does not like cats. I think he does not like cats.",​ "He is a dog person."]​
genbit_metrics_object.add_data(test_text, tokenized=False)
```

This process can be repeated as needed. Every separate sting will be treated as an individual document, i.e. the context window will not reach beyond the limits of a single string. Therefore if a document is coherent, the content should be appended and added as a single string in the input list.

The metric works best on bigger corpora, we suggest analyzing at least 400 documents and 600 unique words (excluding stop words).

To generate the gender bias metrics, we run `get_metrics` by setting `output_statistics` and `output_word_lists` tp false, we can reduce the number of metrics created.

```
metrics = genbit_metrics_object.get_metrics(output_statistics=True, output_word_list=True)
```

## Metrics

GenBit computes a number of metrics, which are functions of word co-occurrences with predefined "gendered" words. These gendered words are divided into "female" words and "male" words. Female words contains items like "her" and "waitress" and male words contains items like "he" and "actor". The main calculation is computing co-occurrences between words `w` in your provided text, and words `f` from the female list and words `m` from the male list. In all that follows, `c[w,F]` (respectively, `c[w,M]`) denotes the frequency that word `w` co-occurs with a word on the female (respectively, male) lists. These are naturally interpretable as probabilities by normalizing: `p(w|F) = c[w,F] / c[.,F]` where `c[.,F]` is the sum over all `w'` of `c[w',F]`.

### Overall Metrics

These overall metrics will always be returned.

- **avg_bias_ratio**: The average of the bias ratio scores per token. Formally, this is the average over all words `w` of `log( c[w,M] / c[w,F] )`.
- **avg_bias_conditional**: The average of the bias conditional ratios per token. Similarly, this is the average over all words `w` of `log( p(w|M) / p(w|F) )`.
- **avg_bias_ratio_absolute**. The average of the absolute bias ratio scores per token. Similar to `avg_bias_ratio`, this is the average over all `w` of `| log( c[w,M] / c[w,F] ) |`.
- **avg_bias_conditional_absolute (genbit_score)**: The average of the absolute bias conditional ratios per token [note: this score is most commonly used as the key bias score in the literature]. Formally, this is the average over all `w` of `| log( p(w|M) / p(w|F) ) |`.
- **avg_non_binary_bias_ratio**: the average of the token-level non-binary bias token ratios. Formally, this is the average over all words `w` of `log( (c[w,M] + c[w,F]) / c[w,NB] )`
- **avg_non_binary_bias_conditional**: the average of the token-level non-binary bias conditional ratios. This is the average over all words `w` of `log( p(w|M) + p(w|F) / p(w|NB) )`
- **avg_non_binary_bias_ratio_absolute**: the average of the token-level absolute non-binary bias ratio scores. Similar to `avg_non_binary_bias_ratio`, this is the average over all `w` of `| log( (c[w,M] + c[w,F]) / c[w,NB] ) |`
- **avg_non_binary_bias_conditional_absolute**: the average of the token-level absolute non-binary bias conditional ratios.
- **std_dev_bias_ratio**: Standard deviation of the bias ratio scores. This is the standard deviation that corresponds to `avg_bias_ratio`.
- **std_dev_bias_conditional**: Standard deviation of the bias conditional scores. This is the standard deviation that corresponds to `avg_bias_conditional`.
- **std_dev_non_binary_bias_ratio**: Standard deviation of non binary bias ratio scores. This is the standard deviation that corresponds to `avg_non_binary_bias_ratio`.
- **std_dev_non_binary_bias_conditional**: Standard deviation of non binary bias conditional scores. This is the standard deviation that corresponds to `avg_non_binary_bias_conditional`.
- **percentage_of_female_gender_definition_words**: The percentage of gendered (male, female and non-binary) words in the corpus that belong to the list of female gendered words
- **percentage_of_male_gender_definition_words**: The percentage of gendered (male, female and non-binary) words in the corpus that belong to the list of male gendered words
- **percentage_of_non_binary_gender_definition_words**: The percentage of gendered (male, female and non-binary words in the corpus that belong to the list of non-binary gendered words)

### Metric Statistics

An optional set of `dict` containing statistics that can be included as part of the metrics dict object by the key `statistics`. These statistics will be returned if `output_statistics=True`.

- **frequency_cutoff**: the percentile frequency cutoff value. Any co-occurrence counts that are above this frequency will be used in calculating the metrics
- **num_words_considered**: the count of words that were included in calculating the metrics
- **freq_of_female_gender_definition_words**: The number of times any of the female gendered words occur in the corpus
- **freq_of_male_gender_definition_words**: The number of times any of the male gendered words occur in the corpus
- **freq_of_non_binary_gender_definition_words**: The number of times any of the non-binary gendered words occur in the corpus
- **jsd**: The Jensen-Shannon divergence between the word probabilities conditioned on male and female gendered words. This is `JSD( p(w|M) || p(w|F) )`, where `JSD(q||p)` is the average KL divergence between `q` and `m`, and between `p` and `m`, where m is the average distribution `(p+q)/2`.

### Token Based Metrcs

A `dict` containing object containing per word bias statistics that can be included as part of the metrics dict object by the key `token_based_metrics`. These statistics will be returned if `output_word_list=True`.
By Token:

- **frequency**: The frequency of the token (the number of times it appears in the document): `c[w,.]`
- **male_count**: the number of times the word occurs within context of a male gendered word: `c[w,M]`
- **female_count**: the number of times the word occurs within context of a female gendered word: `c[w,F]`
- **non_binary_count**: the number of times the word occurs within context of a male gendered word: `c[w,NB]`
- **female_conditional_prob**: the conditional probability of the token occurring in context with a female gendered word: `p(w|F)`
- **male_conditional_prob**: the conditional probability of the token occurring in context with a male gendered word: `p(w|M)`
- **bias_ratio**: log(male_count / female_count) the more positive the value, the more biased the word is towards being associated with male gendered word, the more negative the value the more biased the word is towards being associated with female gendered. Each value is `log( c[w,M] / c[w,F] )`. A value of zero means that this word is equally likely to appear co-occurring with words in the female list as the male list (positive indicates more co-occurrence with male words; negative indicates more co-occurrence with female words).
- **non_binary_bias_ratio**: log((male_count + female_count) / non_binary_count) the more positive the value, the more biased the word is towards being associated with a binary gendered word. A value of zero means the word has non-binary gender bias associated with it
- **bias_conditional_ratio**: log( male_cond_prob/female_cond_prob ) the more positive the value, the more biased the word is towards being associated with male gendered word, the more negative the value the more biased the word is towards being associated with female gendered. Each value is `log( p(w|M) / p(w|F) )`. A value of zero means that the probability of this word co-occurring with words in the female list is equal to the probability of co-occurring with words in the male list (positive indicates more likely co-occurrence with male words; negative indicates more likely co-occurrence with female words).
- **non_binary_bias_conditional_ratio**: log( (male_cond_prob+female_cond_prob) / non_binary_cond_prob ) the more positive the value, the more biased the word is towards being associated with binary gendered words; the more negative the value the more biased the word is towards being associated with a non-binary gendered word. A value of zero means the word has no gender bias associated with it.

### Metric Scores, Benchmarking and Analysis

A detailed benchmarking was conducted to evaluate Genbit's performance across different samples and bias condiction in the corpora/datasets.

The score interpretation depends on two key factor,

a) The percentage of male or female gendered definition words

b) Average bias condition absolute score(Genbit Score)

It is observed that With the increase in the gendered definition word percentage the Genbit Score tends to surge, demonstrating the presence of gender bias.

A detailed benchmarking is conducted to study the correlation of score ranges across different datasets and to examine how genderbias could influence the overall machine learning task using multilingual parallel datasets[Winogender-schema, WinoMT, Curated-WinoMT, IMDB, TedTalks and few others].

**Table1: Genbit V2 Reference Score Range for biased datasets.**
| Language | Score Range | Data Size | Bias % Indicator<br>(moderate-high) |
|:--------: |------------- |:------------: |:-----------------------------------: |
| EN | 0.30-1.0+ | >400 Samples | > 0.30 |
| IT | 0.50-1.5+ | >400 Samples | > 1.00 |
| DE | 0.60-2.4+ | >200 Samples | > 0.60 |
| ES | 0.60-2.5+ | >400 Samples | > 0.60 |
| FR | 0.50-1.3+ | >200 Samples | > 0.60 |
| RU | 0.80-2.3+ | >400 Samples | > 1.10+ |

**Note**: The score ranges are derived from certain type of datasets and may vary with datasets. The bias indicator percentage can aid in understanding the degree of biased a dataset can be. A genbit score of greater than the value provided in the last column indicates observable gender bias in the data set that may impact any resulting model trained on the dataset negatively (we would dub this 'moderate' gender bias). The higher this value the great the gender bias in the dataset. It is recommended as a best practive to use both the **genbit_score** as well as observe the values given for **percentage_of_male/female/non-binary_gender_definition_words** to provide some indication of the reliability of the **genbit_score**. In a 'naturally' distributed dataset you would expect that the percentage values for the male/female/non-binary gender definition words not to be overly skewed e.g. if the value observed was 10% male_gender_definition_words, 90% female_gender_definition_words, 0% non-binary_gender_definition_words this would potentially indicate quality conerns with the dataset as such a extreme skew is unlikely (and definitely undesirable) in a dataset.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
