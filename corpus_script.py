import pickle
import pandas as pd

from utils.logger import logger
from utils.timeless import TimeLess

from LDA.lda_trainer import LDATrainer

processed_tweets = []
max_workers = 4

trainer = LDATrainer()
timey = TimeLess()

# Defining data types of ambiguous columns
dtypes = {"in_reply_to_userid": "str", "retweet_userid": "str"}
# Loading tweets data frame un chunks due to its size.
eg_tweets = pd.read_csv(
    "data/egypt_tweets_2020/egypt_022020_tweets_csv_hashed.csv",
    dtype=dtypes,
    chunksize=500000,
    encoding="utf-8",
)

logger.info("Loaded tweets dataframe.")
logger.info("Tweets processing initiated.")

# Execute Tokenization task.
timey.execute_parallel(
    max_workers, eg_tweets, "Tokenize", trainer.process_text, 0, "tweet_text", True
)

# Extend processed keywords list.
for future in timey.futures["Tokenize"]:
    processed_tweets.extend(future.result())

# Dump processed token
for i in range(1, 17):
    with open(f"process_tokens_chunk_{i}.pkl", "rb") as f:
        temp = pickle.load(f)
        processed_tweets.extend(temp)

# Delete unnecessary future results.
timey.delete_futures("Tokenize")

# Build bi-grams corpus.
logger.info("Tweets processing finished, building bi-grams model.")
bigram_mod = trainer.bi_grams_model(processed_tweets)
bigram = [bigram_mod[tweet] for tweet in processed_tweets]

with open("bigram_train.pkl", "wb") as f:
    pickle.load(f)
    logger.info("Dumped bigrams.")


logger.info("Creating id-to-word dictionary.")
timey.execute_parallel(
    max_workers=2,
    num_chunks=16,
    df=bigram,
    task_name="ID Dict",
    func_handler=trainer.id2word_constructor,
)
id2word = trainer.merge_dicts(timey.futures["ID Dict"])
with open("id2word_train.pkl", "wb") as f:
    pickle.dump(id2word, f)
    logger.info("Dumped id2words.")


logger.info("Filtering extreme word counts.")
id2word.filter_extremes(no_below=5, no_above=0.35)
id2word.compactify()
corpus = [id2word.doc2bow(text) for text in bigram]

with open("train_corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)
    logger.info("Dumped corpus.")
