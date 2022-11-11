# GenBit: A Tool for Measuring Gender Bias in Text Corpora

This Responsible-AI-Toolbox-GenBit repo consists of a python library that aims to empower data scientists and ML developers to measure gender bias in their Natural Language Processing (NLP) datasets. 

This repo is a part of the [Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox#responsible-ai-toolbox), a suite of tools providing a collection of model and data exploration and assessment user interfaces and libraries that enable a better understanding of AI systems. These interfaces and libraries empower developers and stakeholders of AI systems to develop and monitor AI more responsibly, and take better data-driven actions.

With the increasing adoption of Natural Language Processing (NLP) models in real-world applications and products, it has become more critical than ever to understand these models' biases and potential harms they could cause to their end users. Many NLP systems suffer from various biases often inherited from the data on which these systems are trained. The prejudice is exhibited at multiple levels spilling from how individuals generate, collect, and label the information leveraged into datasets. Datasets, features, and rules in machine learning algorithms absorb and often magnify such biases present in datasets. Therefore, it becomes essential to measure preferences at the data level to prevent unfair model outcomes.

This repository introduces the Gender Bias Tool (**G**en**B**i**t**), a tool to measure gender bias in NLP datasets. The main goal of GenBit is to analyze your corpora and compute metrics that give insights into the gender bias present in a corpus. The computations in this tool are based primarily on ideas from Shikha Bordia and Samuel R. Bowman, "[Identifying and reducing gender bias in word-level language models](https://arxiv.org/abs/1904.03035)" in the NAACL 2019 Student Research Workshop. 

GenBit helps determine if gender is uniformly distributed across data by measuring the strength of association between a pre-defined list of gender definition words and other words in the corpus via co-occurrence statistics. The key metric it produces (the genbit_score) gives an estimate of the strength of association, on average, of any word in the corpus with a male, female, non-binary, transgender (trans), and cisgender (cis) gender definition words. The metrics that it provides can be used to identify gender bias in a data set to enable the production and use of more balanced datasets for training, tuning and evaluating machine learning models. It can also be used as a standalone corpus analysis tool.


GenBit supports 5 languages: English, German, Spanish, French, Italian and Russian. For English it provides metrics for both binary, non-binary, transgender, and cisgender bias; for the remaining four languages we currently only support binary gender bias. To deal with the challenges of grammatical gender in non-English languages, it leverages [stanza lemmatizers](https://stanfordnlp.github.io/stanza/lemma.html). It also uses the NLTK tokenization libraries. The full list of requirements are listed in [requirements.txt](requirements.txt)

## Contents
- [Install GenBit](#installation)
- [Use GenBit](#use)
- [Gendered Terms](#terms)
- [Supported Metrics](#metrics)
- [Metric Scores, Benchmarking and Interpretation](#interpret)
- [Citation](#citation)
- [Useful Links](#links)
- [Contributing](#contributing)
- [Trademarks](#trademarks)
- [Authors and acknowledgment](#authors)

# <a name="installation"></a>
## Install GenBit

The package can be installed from [pypi](https://pypi.org/project/genbit/) with:

```
pip install genbit
```

Tested and supported environments for the GenBit python package are:
- Local usage on Windows
- Local usage on Linux (Tested on Ubuntu 18.04 and Debian Buster 10)
- As part of Azure functions installed using a [remote build](https://docs.microsoft.com/en-us/azure/azure-functions/functions-reference-python#remote-build-with-extra-index-url).



# <a name="use"></a>
## Use GenBit

To use GenBit metrics, run the following code:

```python
from genbit.genbit_metrics import GenBitMetrics

# Create a GenBit object with the desired settings:

genbit_metrics_object = GenBitMetrics(language_code, context_window=5, distance_weight=0.95, percentile_cutoff=80)


# Let's say you want to use GenBit with a test sentence, you can add the sentence to GenBit:
test_text = ["I think she does not like cats. I think he does not like cats.", "He is a dog person."]

genbit_metrics_object.add_data(test_text, tokenized=False)


# To generate the gender bias metrics, we run `get_metrics` by setting `output_statistics` and `output_word_lists` to false, we can reduce the number of metrics created.


metrics = genbit_metrics_object.get_metrics(output_statistics=True, output_word_list=True)

```

This process can be repeated as needed. Every separate string will be treated as an individual document, i.e. the context window will not reach beyond the limits of a single string. Therefore if a document is coherent, the content should be appended and added as a single string in the input list.

The metric works best on bigger corpora; therefore, we suggest analyzing at least 400 documents and 600 unique words (excluding stop words).


# <a name="terms"></a>
## Gendered Terms
We have collected a [list of "gendered" terms for female, male, non-binary, binary, transgender, and cisgender groups](https://github.com/microsoft/responsible-ai-toolbox-genbit/tree/main/genbit/gendered-word-lists).

- [Female words](https://github.com/microsoft/responsible-ai-toolbox-genbit/blob/main/genbit/gendered-word-lists/en/female.txt) contain terms such as "her" and "waitress".
- [Male words](https://github.com/microsoft/responsible-ai-toolbox-genbit/blob/main/genbit/gendered-word-lists/en/male.txt) contain terms such as "he" and "fireman".
- [Non-binary words](https://github.com/microsoft/responsible-ai-toolbox-genbit/blob/main/genbit/gendered-word-lists/en/non-binary.txt) contain terms such as "sibling" and "parent".
- [Cisgender words](https://github.com/microsoft/responsible-ai-toolbox-genbit/blob/main/genbit/gendered-word-lists/en/cis.txt) contain terms such as "cisgender" and "cissexual".
- [Transgender words](https://github.com/microsoft/responsible-ai-toolbox-genbit/blob/main/genbit/gendered-word-lists/en/trans.txt) contain terms such as "trans woman" and "trans man".

Please note that there is no explicit file for binary terms because the terms from male.txt and female.txt will be combined to form the binary word list.

The inclusion criterion for terms that are not inherently gendered (e.g., "womb" or "homemaker" in female.txt) should be: If the association of the term with a particular gender is only due to a social phenomenon that we want to approximate with our measurement (e.g., stereotyping, discrimination), then the term should be excluded. The categories are defined in terms of similar societal stereotypes/discrimination/bias/stigma because that is what GenBit would like to measure. 

### Creation of New Word Lists for Male/Female
The gender definition terms consist of pairs of corresponding entries – one for male gendered forms and one for female gendered forms. In general, for every male gendered form, there should be one or more female gendered forms, and for every female gendered form there should be one or more male gendered form. It should be noted, however, that there may be some rare cases, where there are entries that do not have a opposite gender equivalent form in some languages.

A gender definition entry is a concept that can only ever be attributed to a person of a specific gender; they represent words that are associated with gender by definition. They are unambiguous – they only ever have one possible semantic interpretation. They are primarily nouns or pronouns that refer to a person of a specific gender (e.g. he/she/zie, father/mother) but may occasionally include adjectives/adverbs/verbs (e.g. masculine / feminine, manly/womanly).


### Creation of New Word Lists for Non-binary/Binary 
The binary/non-binary dimension has 3 categories: 
- Binary: terms that typically refer to men or women, but not to non-binary people 
- Non-binary: terms that specifically refer to non-binary people and that are not typically used to refer to men or women - these terms require abandoning the idea of there being only two genders 
- All-gender: terms that can refer to people of any gender, including men, women, and non-binary people 

<b>Motivation for this split</b>: If we just put both non-binary and all-gender terms in the same txt file, we will not get any meaningful numbers out of GenBiT because the all-gender terms are so much more frequent and they're often used to refer to binary people, so we wouldn't really be measuring the anti-non-binary bias that we would like to measure.  On the other hand, if we only use the explicitly non-binary terms, we'll miss a lot of textual references to non-binary people that we would catch for binary people (e.g., we'd catch "brother" and "sister", but not "sibling"). 

We combined the categories “non-binary” and “all-gender” into one word list. 


### Creation of New Word Lists for Transgender/Cisgender

The transgender category includes all terms that relate to transgressing gender boundaries or that are perceived as transgressing gender boundaries (for example, as evidenced by the type of stigma they receive). This is why the trans category includes terms for people that may not self-identify as trans, but that also transgress gender boundaries. 

All terms in the non-binary category are also included in the trans category. Terms that are in the trans category, but not in the non-binary category, are terms that refer to transgressing gender boundaries, but that doesn't necessarily require abandoning the idea of there only being two genders. 


The collected terms could be found [here](https://github.com/microsoft/responsible-ai-toolbox-genbit/tree/main/genbit/gendered-word-lists), and you can explore [Lexicon Guidelines](https://github.com/microsoft/responsible-ai-toolbox-genbit/blob/main/genbit/gendered-word-lists/LEXICON_GUIDELINES.md) for creating lexicons to support new languages.



# <a name="metrics"></a>
## Metrics

With the predefined "gendered" terms collected and available for you, GenBit computes a number of metrics which are functions of word co-occurrences with these predefined "gendered" words. As a reminder, the gendered words are divided into "female" words, "male" words, "trans" words (English only), "cis" words (English only), and "non-binary" words (English only). 

### Female vs Male Calculations
The main calculation is computing co-occurrences between words `w` in your provided text, and words `f` from the female list and words `m` from the male list. In all that follows, `c[w,F]` (respectively, `c[w,M]`) denotes the frequency that word `w` co-occurs with a word on the female (respectively, male) lists. These are naturally interpretable as probabilities by normalizing: `p(w|F) = c[w,F] / c[.,F]` where `c[.,F]` is the sum over all `w'` of `c[w',F]`.

### Non-binary vs Binary Calculations (English only)
The main calculation is computing co-occurrences between words `w` in your provided text, and words `nb` from the non-binary list and words `b` from the binary list. In all that follows, `c[w,nb]` (respectively, `c[w,b]`) denotes the frequency that word `w` co-occurs with a word on the binary (respectively, non-binary) lists. These are naturally interpretable as probabilities by normalizing: `p(w|nb) = c[w,nb] / c[.,nb]` where `c[.,nb]` is the sum over all `w'` of `c[w',nb]`.

### Transgender vs Cisgender Calculations (English only)
The main calculation is computing co-occurrences between words `w` in your provided text, and words `t` from the transgender list and words `c` from the cisgender list. In all that follows, `c[w,t]` (respectively, `c[w,t]`) denotes the frequency that word `w` co-occurs with a word on the transgender (respectively, cisgender) lists. These are naturally interpretable as probabilities by normalizing: `p(w|t) = c[w,t] / c[.,t]` where `c[.,t]` is the sum over all `w'` of `c[w',t]`.

### Supported Metrics

GenBit supports the following dataset metrics:
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

# <a name="interpret"></a>
## Metric Scores, Benchmarking and Interpretation

A detailed benchmarking was conducted to evaluate Genbit's performance across different samples and quantities of gender bias in the corpora/datasets.

The score interpretation depends on two key factors,

a) The percentage of male or female gendered definition words

b) Average bias condition absolute score (genbit_score)

It is observed that With the increase in the gendered definition word percentage the Genbit Score tends to surge, demonstrating the presence of gender bias.

A detailed benchmarking is conducted to study the correlation of score ranges across different datasets and to examine how genderbias could influence the overall machine learning task using multilingual parallel datasets[Winogender-schema, WinoMT, Curated-WinoMT, IMDB, TedTalks and few others].

**Table1: GenBit V2 Reference Score Range for biased datasets.**
| Language | Score Range | Data Size | Bias % Indicator<br>(moderate-high) |
|:--------: |------------- |:------------: |:-----------------------------------: |
| EN | 0.30-1.0+ | >400 Samples | > 0.30 |
| IT | 0.50-1.5+ | >400 Samples | > 1.00 |
| DE | 0.60-2.4+ | >200 Samples | > 0.60 |
| ES | 0.60-2.5+ | >400 Samples | > 0.60 |
| FR | 0.50-1.3+ | >200 Samples | > 0.60 |
| RU | 0.80-2.3+ | >400 Samples | > 1.10+ |

The score ranges are derived from certain type of datasets and may vary with datasets. The bias indicator percentage can aid in understanding the degree of biased a dataset can be. A genbit score of greater than the value provided in the last column indicates observable gender bias in the data set that may impact any resulting model trained on the dataset negatively (we would dub this 'moderate' gender bias). The higher this value the great the gender bias in the dataset. 

It is recommended as a best practice to use both the **genbit_score** as well as observe the values given for **percentage_of_male/female/non-binary_gender_definition_words** to provide some indication of the reliability of the **genbit_score**. In a 'naturally' distributed dataset you would expect that the percentage values for the male/female/non-binary gender definition words not to be overly skewed e.g. if the value observed was 10% male_gender_definition_words, 90% female_gender_definition_words, 0% non-binary_gender_definition_words this would potentially indicate quality concerns with the dataset as such a extreme skew is unlikely (and definitely undesirable) in a dataset. 

# <a name="citation"></a>
## Citation
<a>
<pre>
@article{sengupta2021genbit,
  title={GenBiT: measure and mitigate gender bias in language datasets},
  author={Sengupta, Kinshuk and Maher, Rana and Groves, Declan and Olieman, Chantal},
  journal={Microsoft Journal of Applied Research},
  year={2021},
  volume={16},
  pages={63--71}}
}
</pre>
</a>
<a href="https://www.microsoft.com/en-us/research/uploads/prod/2021/10/MSJAR_Genbit_Final_Version-616fd3a073758.pdf">Paper link</a>
  
  
# <a name="links"></a>
## Useful Links

+ [Get started](notebooks/quickstart_sample_notebook.ipynb) using a sample Jupyter notebook.
+ [Identifying and Reducing Gender Bias in Word-Level Language Models](https://arxiv.org/pdf/1904.03035.pdf): Bordia and Bowman paper that describes the approach that GenBit is based on.
+ [Winogender](https://github.com/rudinger/winogender-schemas) Winogender data set; we use samples from these dataset as part of the GenBit tests and in our sample Jupyter notebook.

# <a name="contributing"></a>
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


# <a name="trademarks"></a>
## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

# <a name="authors"></a>
## Authors and acknowledgment

The original GenBit tool was co-authored by (listed in alphabetical order) Declan Groves, Chantal Olieman, Kinshuk Sengupta, David Riff, Eshwar Stalin, Marion Zepf.

The team members behind the open source release of the tool are (listed in alphabetical order) Chad Atalla, Hal Daumé III, Declan Groves, Mehrnoosh Sameki, Kinshuk Sengupta, and Marion Zepf.
