import json
from typing import Callable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding


class LaMPDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        label_path: str,
        prompt_generator: Callable[[str, list[dict[str, str]], float], str] | None = None
    ) -> None:
        with open(data_path, 'r') as file:
            data = json.load(file)

        with open(label_path, 'r') as file:
            label_dict = json.load(file)
            labels = {label['id']: label['output'] for label in label_dict['golds']}

        self.data = data
        self.labels = labels
        self.prompt_generator = prompt_generator

    def __getitem__(self, index: int) -> dict[str, str]:
        example = self.data[index]
        source = example['input']

        if self.prompt_generator is not None:
            source = self.prompt_generator(source, example['profile'])

        return {
            'id': example['id'],
            'source': source,
            'target': self.labels[example['id']] 
        }

    def __len__(self) -> int:
        return len(self.data)


class RetrieverTrainingDataset(Dataset):

    def __init__(self, data_path: str, label_path: str, query_corpus_generator: Callable) -> None:
        super().__init__()

        with open(data_path, 'r') as file:
            self.data = json.load(file)

        with open(label_path, 'r') as file:
            label_dict = json.load(file)
            self.labels = {label['id']: label['output'] for label in label_dict['golds']}

        self.query_corpus_generator = query_corpus_generator

    def __getitem__(self, index: int) -> dict[str, str | list[str]]:
        example = self.data[index]

        source = example['input']
        profile = example['profile']
        query, corpus = self.query_corpus_generator(source, profile)

        return {
            'id': example['id'],
            'source': source,
            'profile': profile,
            'query': query,
            'corpus': corpus,
            'target': self.labels[example['id']]
        }

    def __len__(self) -> int:
        return len(self.data)


class RetrieverTrainingCollator:

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_n_profiles: int,
        max_query_length: int,
        max_document_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_n_profiles = max_n_profiles
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length

    def __call__(self, examples: list[dict[str, str | list[str]]]) -> (
        dict[str, list[str] | list[list[str]] | BatchEncoding | torch.Tensor]
    ):
        ids = []
        sources = []
        profiles = []
        queries = []
        corpuses = []
        targets = []

        for example in examples:
            ids.append(example['id'])
            sources.append(example['source'])
            profiles.append(example['profile'])
            queries.append(example['query'])
            corpuses.append(example['corpus'])
            targets.append(example['target'])

        # Keep only `self.max_n_profiles` profiles for each example
        max_n_profiles = max(len(profile) for profile in profiles)

        if self.max_n_profiles > 0:
            max_n_profiles = min(max_n_profiles, self.max_n_profiles)

        profile_mask = torch.ones(len(examples), max_n_profiles, dtype=torch.bool)

        for batch_index, corpus in enumerate(corpuses):
            if len(corpus) < max_n_profiles:
                profile_mask[batch_index, len(corpus):] = 0
                corpus.extend([''] * (max_n_profiles - len(corpus)))
            elif len(corpus) > max_n_profiles:
                corpus[max_n_profiles:] = []
                profiles[batch_index][max_n_profiles:] = []

        query_inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors='pt'
        )

        all_corpus_inputs = []
        all_documents = [document for corpus in corpuses for document in corpus]

        for documents in [
            all_documents[index : index + 100]
            for index in range(0, len(all_documents), 100)
        ]:
            corpus_inputs = self.tokenizer(
                documents,
                padding=True,
                truncation=True,
                max_length=self.max_document_length,
                return_tensors='pt'
            )
            all_corpus_inputs.append(corpus_inputs)

        return {
            'source': sources,
            'profile': profiles,
            'query_inputs': query_inputs,
            'all_corpus_inputs': all_corpus_inputs,
            'profile_mask': profile_mask,
            'target': targets
        }
