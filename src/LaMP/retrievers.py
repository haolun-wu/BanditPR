import random
from typing import Callable

from rank_bm25 import BM25Okapi

import torch
from transformers import AutoTokenizer, AutoModel


def create_retriever(retriever: str, device: str | None = None) -> (
    Callable[[str, list[dict[str, str]], int, Callable], list[str]]
):
    if retriever == 'contriever':
        contriever = _ContrieverRetriever()
        contriever.to(device)
        return contriever

    retriever_fns = {
        'random': _random_retriever,
        'bm25': _bm25_retriever
    }
    return retriever_fns[retriever]


def _random_retriever(input_, profiles, n_retrieve, query_corpus_generator):
    n_retrieve = min(n_retrieve, len(profiles))
    retrieved_profiles = random.choices(profiles, k=n_retrieve)
    return retrieved_profiles


def _bm25_retriever(input_, profiles, n_retrieve, query_corpus_generator):
    n_retrieve = min(n_retrieve, len(profiles))
    query, corpus = query_corpus_generator(input_, profiles)

    tokenized_query = query.split()
    tokenized_corpus = [document.split() for document in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    retrieved_profiles = bm25.get_top_n(tokenized_query, profiles, n=n_retrieve)
    return retrieved_profiles


class _ContrieverRetriever:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.contriever = AutoModel.from_pretrained('facebook/contriever')
        self.contriever.eval()

    def to(self, device):
        self.contriever.to(device)

    @torch.no_grad()
    def __call__(self, input_, profiles, n_retrieve, query_corpus_generator):
        n_retrieve = min(n_retrieve, len(profiles))
        query, corpus = query_corpus_generator(input_, profiles)

        scores = []
        query_embedding = self._compute_sentence_embeddings(query)

        for batch_corpus in [corpus[i : i + 4] for i in range(0, len(corpus), 4)]:
            batch_corpus_embeddings = self._compute_sentence_embeddings(batch_corpus)
            batch_scores = (query_embedding @ batch_corpus_embeddings.T).squeeze(dim=0)
            scores.append(batch_scores)

        scores = torch.cat(scores, dim=0)

        _, indices = torch.topk(scores, n_retrieve, dim=0)
        retrieved_profiles = [profiles[index] for index in indices]
        return retrieved_profiles

    def _compute_sentence_embeddings(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to(self.contriever.device)

        outputs = self.contriever(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(dim=-1)

        token_embeddings = token_embeddings.masked_fill(attention_mask == 0, 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        return sentence_embeddings
