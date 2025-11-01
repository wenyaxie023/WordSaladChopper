from skythought.evals.util.math_parsing_util import (
    extract_answer,
    math_equal,
    strip_answer_string,
)

from ..base import TaskHandler
from pipeline.config import PipelineConfig
from pipeline.utils import load_json

class MathTaskHandler(TaskHandler):
    def __init__(self, cfg: PipelineConfig, params: dict | None = None):
        super().__init__(cfg, params)

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
        return self.task_config.templating_parameters["template"].format(**problem)

    def check_correctness(self, problem, generation):
        # answer = strip_answer_string(problem[self.task_config.answer_key])
        answer = strip_answer_string(problem["answer"])
        pred = extract_answer(generation)
        print(f"pred: {pred}")
        print(f"answer: {answer}")
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)

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

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None
    ):
        dataset = self.load_dataset(subset=subset, split=split).to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]
