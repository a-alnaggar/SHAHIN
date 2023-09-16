import gensim

from gensim.models.phrases import Phraser
from typing import List


from processor import ArProcessor


class LDATrainer:
    def __init__(self):
        ar_processor = ArProcessor()

    @staticmethod
    def bi_grams_model(words: List[List[str]], bi_min: int = 15) -> Phraser:
        """
        Creates a bi-gram model of tokens of bigrams for LDA model training.
        :param words: List of tokenized words.
        :param bi_min: Threshold for phrases detection.
        :return: Bi-gram phrases model
        """
        bi_gram = gensim.models.Phrases(words, min_count=bi_min)
        bi_gram_mod = Phraser(bi_gram)
        return bi_gram_mod

    # def get_corpus(self, df, text_col):
    #     df["text"] = strip_newline(df.text)
    #     words = list(sent_to_words(df.text))
    #     words = remove_stopwords(words)
    #     bigram_mod = bigrams(words)
    #     bigram = [bigram_mod[review] for review in words]
    #     id2word = gensim.corpora.Dictionary(bigram)
    #     id2word.filter_extremes(no_below=10, no_above=0.35)
    #     id2word.compactify()
    #     corpus = [id2word.doc2bow(text) for text in bigram]
    #     return corpus, id2word, bigram
