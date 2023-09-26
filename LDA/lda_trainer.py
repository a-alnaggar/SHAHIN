import gensim
import pickle

from typing import List
from pandas import DataFrame
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from concurrent.futures import Future

from utils.logger import logger
from utils.processor import ArProcessor


class LDATrainer:
    """
    Constructs and preprocess LDA training data.
    """

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

    def process_text(
        self,
        df: DataFrame,
        chunk_num: int = 0,
        text_col: str = "text",
        cache: bool = False,
    ) -> List[str]:
        """
        Process text for building LDA corpora.
        :param df: Dataframe object.
        :param text_col: Column name that contains text to process.
        :param chunk_num: Number of dataframe chunk.
        :param cache: Cache after processing.
        :return: Nested List of tokenized words.
        """
        if chunk_num:
            logger.info("Started processing chunk %s", chunk_num)

        processed_col = df[text_col].apply(
            lambda row: self.ar_processor.preprocess_arabic_text(row, tokenized=True)
        )
        if chunk_num:
            logger.info("Finished processing chunk %s", chunk_num)

        if cache:
            with open(f"process_tokens_chunk_{chunk_num}.pkl", "wb") as fs:
                pickle.dump(processed_col.to_list(), fs)

        return processed_col.to_list()

    @staticmethod
    def id2word_constructor(tokens: List[str], dict_num: int) -> Dictionary:
        """
        Creates a Dictionary Object.
        :param tokens: List of tokens
        :param dict_num: Dictionary number for logging.
        :return: Dictionary
        """
        id2word_dict = Dictionary(tokens)
        logger.info("Constructed id-to-word dictionary number: %s", dict_num)
        return id2word_dict

    @staticmethod
    def merge_dicts(dicts: List[Future]) -> Dictionary:
        """
        Merges all futures `Dictionary` objects.
        :param dicts: List of finished futures.
        :return: Combined Dictionary object.
        """
        if len(dicts) == 1:
            return dicts[0].result()

        init_dict = Dictionary()

        for future in dicts:
            init_dict.add_documents(future.result())

        return init_dict
