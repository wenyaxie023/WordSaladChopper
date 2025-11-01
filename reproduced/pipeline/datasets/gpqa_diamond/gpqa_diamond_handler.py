import random

from skythought.evals.util.math_parsing_util import get_multiple_choice_answer

from ..base import TaskHandler
from pipeline.config import PipelineConfig
from pipeline.utils import load_json
import re

class GPQADiamondTaskHandler(TaskHandler):
    def __init__(self, cfg: PipelineConfig, params: dict | None = None):
        super().__init__(cfg, params)
        self.params = params

        # ↓ prefer local path if provided
        if "path" in self.params:
            import json, pathlib
            data = load_json(pathlib.Path(self.params["path"]))
            ## just select 10 for debugging
            ## data is a dict of dict
            data = {k: v for k, v in list(data.items())}
            self.ds = data

        else:
            # self.ds = datasets.load_dataset(self.HF_REPO, split="test")
            raise NotImplementedError("Dataset loading not implemented")
    def generate_prompt(self, problem):
        multiple_choice_string, correct_answer_letter = (
            self.get_multiple_choice_answers(problem)
        )
        problem["Answer"] = correct_answer_letter
        problem["prompt"] = problem["Question"] + "\n" + multiple_choice_string
        return self.task_config.templating_parameters["template"].format(
            prompt=problem["prompt"]
        )

    def update_results(self, problem, response):
        # Initialize the response structure
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }
        curr_res = self.check_correctness(problem, generation=response)
        if curr_res:
            response_entry["correctness"] = True
            response_entry["reason"] = ""
        else:
            response_entry["correctness"] = False
            response_entry["reason"] = "Solution is incorrect."

        return response_entry

    def check_correctness(self, problem, generation):
        if "</think>" in generation:
            last_think_index = generation.rfind("</think>")
            generation = generation[last_think_index + len("</think>"):]
        pred = get_multiple_choice_answer(generation)
        answer = problem["Answer"]
        return answer == pred

    def get_multiple_choice_answers(self, data):
        answers = [
            data["Correct Answer"],
            data["Incorrect Answer 1"],
            data["Incorrect Answer 2"],
            data["Incorrect Answer 3"],
        ]
        random.shuffle(answers)

        # Map options to letters
        options = ["A", "B", "C", "D"]
        options_to_answers = {
            letter: answer for letter, answer in zip(options, answers)
        }

        # Format the options into the string
        multiple_choice_string = ", ".join(
            f"{letter}) {options_to_answers[letter]}" for letter in options
        )

        # Save the letter corresponding to the correct answer
        correct_answer_letter = next(
            letter
            for letter, answer in options_to_answers.items()
            if answer == data["Correct Answer"]
        )

        return multiple_choice_string, correct_answer_letter

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None
    ):
        train_data = self.load_dataset(subset=subset, split=split).to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]
