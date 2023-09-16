import gensim

from gensim.models.phrases import Phraser
from pandas import DataFrame
from typing import List


from utils.processor import ArProcessor


class LDATrainer:
    def __init__(self):
        self.ar_processor = ArProcessor()

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

    def process_text(self, df: DataFrame, text_col: str) -> List[str]:
        """
        Process text for building LDA corpora.
        :param df: Dataframe object.
        :param text_col: Column name that contains text to process.
        :return: Nested List of tokenized words.
        """
        processed_col = df[text_col].apply(
            self.ar_processor.preprocess_arabic_text, args=True
        )
        return processed_col.to_list()

    # def get_corpus(self, df, text_col):
    #     tokens = []
    #     df[text_col] = strip_newline(df.text)
    #     words = list(sent_to_words(df.text))
    #     words = remove_stopwords(words)
    #     bigram_mod = bigrams(words)
    #     bigram = [bigram_mod[review] for review in words]
    #     id2word = gensim.corpora.Dictionary(bigram)
    #     id2word.filter_extremes(no_below=10, no_above=0.35)
    #     id2word.compactify()
    #     corpus = [id2word.doc2bow(text) for text in bigram]
    #     return corpus, id2word, bigram
