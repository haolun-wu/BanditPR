import os
import logging
from typing import TypeAlias

import torch
from torch.utils.data import Dataset
from transformers import pipeline
from transformers.pipelines.text_generation import Chat
from openai import OpenAI, OpenAIError

from tqdm import tqdm

from lamp import get_labels


logger = logging.getLogger(__name__)
Message: TypeAlias = list[dict[str, str]]


SYSTEM_PROMPTS = {
    'LaMP-1': (
        f'You are a personalized citation identification chatbot '
        f'who responds with one of the following: {get_labels("LaMP-1")} based on the given examples.'
    ),
    'LaMP-2': (
        f'You are a personalized movie tagging chatbot '
        f'who responds with one of the following: {get_labels("LaMP-2")} based on the given examples.'
    ),
    'LaMP-3': (
        f'You are a personalized product rating chatbot '
        f'who responds with one of the following: {get_labels("LaMP-3")} based on the given examples.'
    ),
    'LaMP-4': (
        f'You are a personalized news headline generation chatbot '
        f'who generates a news headline in a style similar to the given examples without any additional text.'
    ),
    'LaMP-5': (
        f'You are a personalized scholarly title generation chatbot '
        f'who generates a scholarly title in a style similar to the given examples without any additional text.'
    ),
    'LaMP-6': (
        f'You are a personalized email subject generation chatbot '
        f'who generates an email subject in a style similar to the given examples without any additional text.'
    ),
    'LaMP-7': (
        f'You are a personalized tweet paraphrasing chatbot '
        f'who paraphrases a tweet in a style similar to the given examples without any additional text.'
    ),
    'LongLaMP-1': (
        f'You are a personalized email completion chatbot '
        f'who completes an email in a style similar to the given examples without any additional text.'
    ),
    'LongLaMP-2': (
        f'You are a personalized abstract generation chatbot '
        f'who generates an abstract in a style similar to the given examples without any additional text.'
    ),
    'LongLaMP-3': (
        f'You are a personalized topic generation chatbot '
        f'who generates a topic in a style similar to the given examples without any additional text.'
    ),
    'LongLaMP-4': (
        f'You are a personalized product review generation chatbot '
        f'who generates a product review in a style similar to the given examples without any additional text.'
    )
}


class LLM:

    def __init__(self, task: str, model: str, provider: str, generate_config: dict, verbose: bool = False) -> None:
        self.task = task
        self.model = model
        self.provider = provider
        self.generate_config = generate_config
        self.verbose = verbose

        if self.provider == 'local':
            self.pipeline = pipeline(
                task='text-generation',
                model=self.model,
                device=(torch.cuda.device_count() - 1),
                torch_dtype=torch.bfloat16
            )
            self.pipeline.tokenizer.padding_side = 'left'

            if self.pipeline.tokenizer.pad_token is None:
                self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token
                self.pipeline.model.generation_config.pad_token_id = self.pipeline.tokenizer.eos_token_id
        elif self.provider == 'openai':
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url='https://api.openai.com/v1')
        else:
            raise ValueError(f'Invalid provider: {self.provider}')

    def __call__(self, prompts: list[str]) -> list[str]:
        if self.provider == 'local':
            return self._generate_local(prompts)
        else:
            return self._generate_api(prompts)

    def _generate_local(self, prompts: list[str]) -> list[str]:
        responses = []
        dataset = _ChatDataset([self._create_message(prompt) for prompt in prompts])

        for output in tqdm(
            self.pipeline(dataset, **self.generate_config),
            desc='Generating responses',
            total=len(dataset),
            disable=(not self.verbose)
        ):
            response = output[0]['generated_text'][-1]['content']
            responses.append(response)

        return responses

    def _generate_api(self, prompts: list[str]) -> list[str]:
        responses = [None] * len(prompts)
        remaining_prompts = set(range(len(prompts)))

        while remaining_prompts:
            for index in tqdm(list(remaining_prompts), desc='Requesting responses', disable=(not self.verbose)):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=self._create_message(prompts[index]),
                        **self.generate_config
                    )
                    responses[index] = completion.choices[0].message.content
                    remaining_prompts.remove(index)
                except OpenAIError as err:
                    logger.error(f'OpenAI API error: {err}', exc_info=True)

        return responses

    def _create_message(self, prompt: str) -> Message:
        message = [
            {'role': 'system', 'content': SYSTEM_PROMPTS[self.task]},
            {'role': 'user', 'content': prompt}
        ]
        return message


class _ChatDataset(Dataset):

    def __init__(self, messages: list[Message]) -> None:
        self.chats = [Chat(message) for message in messages]

    def __len__(self) -> int:
        return len(self.chats)

    def __getitem__(self, index: int) -> Chat:
        return self.chats[index]
