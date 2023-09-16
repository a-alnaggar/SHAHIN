import string
import re
import nltk

from pandas import DataFrame
from pandas import Series
from nltk.corpus import stopwords

from typing import Set, Union, List


class ArProcessor:
    """
    Class for processing and normalizing arabic text.
    """

    def __init__(
        self, stopwords_path="/home/naggar/repos/Optimus/EDA/data/stop_words_arabic.txt"
    ):
        self.ar_stopwords = (
            self.read_arabic_stop_words(stopwords_path) if stopwords_path else []
        )

    @staticmethod
    def remove_punctuation(text: str) -> str:
        """
        Remove punctuation from text.
        """
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    @staticmethod
    def read_arabic_stop_words(file_path: str) -> Set[str]:
        """
        Reads a text file of stop words
        :param file_path: Stop words text file path.
        :return: Set of stop words.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            stop_words = {word.strip() for word in file}
        return stop_words

    def preprocess_arabic_text(
        self, text: str, tokenized: bool = False
    ) -> Union[str, List[str]]:
        """
        Preprocessing pipeline for arabic text.
        remove punctuation -> normalize letters -> tokenize -> omit stopwords
        """

        # Remove punctuation
        text = self.remove_punctuation(text)

        # Normalize Arabic characters (optional, depends on the use case)
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        # text = re.sub("[^\u0600-\u06FF\s]", "", text)
        text = re.sub(
            "[^\u0600-\u06FF\u0660-\u0669\u06F0-\u06F9\u0020-\u007E]", "", text
        )
        text = re.sub(r"(\w)[{0}]+(\w)".format(string.punctuation), r"\1 \2", text)

        # Tokenize the text into words (optional, depends on the use case)
        words = nltk.word_tokenize(text)

        english_stop_words = set(stopwords.words("english"))

        # Load Arabic stop words
        arabic_stop_words = self.read_arabic_stop_words("data/stop_words_arabic.txt")

        # Combine both stop word sets
        stop_words = arabic_stop_words.union(english_stop_words)

        # Filter out stop words
        filtered_words = [
            word for word in words if word.lower() not in stop_words and len(word) > 1
        ]
        # Return is tokens if specified
        if tokenized:
            return filtered_words

        # Join the filtered words back into a single string
        cleaned_text = " ".join(filtered_words)

        return cleaned_text

    def get_count(self, chunk: DataFrame, col: str) -> Series:
        """
        Extracts word counts from a dataframe column.
        :param chunk: Dataframe chunk.
        :param col: Column name.
        :return: Pandas series of word counts.
        """
        # Apply Arabic text preprocessing to the 'tweet_text'
        chunk[col] = chunk[col].apply(self.preprocess_arabic_text)

        # Perform word frequency analysis on cleaned text
        word_freq = chunk[col].str.split(expand=True).stack().value_counts()
        return word_freq
