import json
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        # Tokenize the corpus (list of words for each doc)
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, k):
        tokenized_query = query.lower().split()
        # Returns the top k documents based on keyword matching
        return self.bm25.get_top_n(tokenized_query, self.corpus, n=k)