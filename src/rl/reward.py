from typing import Callable

import evaluate


def create_reward_function(task: str) -> Callable[[list[str], list[str]], list[float]]:
    """Create the reward function for the specified task.

    Args:
        task (str): The LaMP task.

    Returns:
        Callable[[list[str], list[str], torch.device], torch.Tensor]:
            The reward function corresponding to the task.
    """
    task_fn = {
        'LaMP-1': classification_reward_function,
        'LaMP-2': classification_reward_function,
        'LaMP-3': regression_reward_function,
        'LaMP-4': create_generation_reward_function(),
        'LaMP-5': create_generation_reward_function(),
        'LaMP-6': create_generation_reward_function(),
        'LaMP-7': create_generation_reward_function()
    }
    return task_fn[task]


def classification_reward_function(predictions: list[str], targets: list[str]) -> list[float]:
    """Compute classification rewards based on prediction and target sequences.

    Args:
        predictions (list[str]): Prediction sequences.
        targets (list[str]): Target sequences.

    Returns:
        rewards (list[float]):
            Rewards indicating whether each prediction matches its target.
    """
    rewards = []

    for prediction, target in zip(predictions, targets):
        reward = float(prediction.strip() == target.strip())
        rewards.append(reward)

    return rewards


def regression_reward_function(predictions: list[str], targets: list[str]) -> list[float]:
    """Compute regression rewards based on prediction and target sequences.

    Args:
        predictions (list[str]): Prediction sequences.
        targets (list[str]): Target sequences.

    Returns:
        reward (list[float]):
            Rewards computed as the negative L1 distance between predictions and targets.
    """
    rewards = []

    for prediction, target in zip(predictions, targets):
        target_value = float(target)

        try:
            prediction_value = float(prediction)
        except ValueError:
            if abs(1 - target_value) > abs(5 - target_value):
                prediction_value = 1.
            else:
                prediction_value = 5.

        reward = -abs(prediction_value - target_value)
        rewards.append(reward)

    return rewards


def create_generation_reward_function() -> Callable[[list[str], list[str]], list[float]]:
    """Wrapper function to initialize the ROUGE metric.

    Returns:
        generation_reward_function (Callable[[list[str], list[str]], list[float]]):
            Function that computes the generation reward.
    """
    rouge_metric = evaluate.load('rouge')

    def generation_reward_function(predictions: list[str], targets: list[str]) -> list[float]:
        """Compute the generation reward based on prediction and target sequences.

        Args:
            predictions (list[str]): Prediction sequences.
            targets (list[str]): Target sequences.

        Returns:
            rewards (list[float]):
                Rewards computed as ROUGE-1 scores for prediction-target pairs.
        """
        rewards = []

        for prediction, target in zip(predictions, targets):
            prediction = [prediction.strip()]
            target = [[target.strip()]]
            rouge_results = rouge_metric.compute(predictions=prediction, references=target)
            rewards.append(rouge_results['rouge1'])

        return rewards

    return generation_reward_function
