from abc import ABC, abstractmethod
from pathlib import Path
import json, random
from typing import Iterable, Dict, List
import re
from datasets import load_dataset, concatenate_datasets
from wscgen.utils import (
    extract_answer,
    math_equal,
    strip_answer_string,
    get_multiple_choice_answer
)

__all__ = ["get_dataset_handler", "DatasetHandler"]

class DatasetHandler(ABC):
    name: str
    prompt_template: str 

    def load(self, root: Path | None = None) -> Iterable[dict]:
        if root is not None:
            yield from self._load_local(Path(root))
            return
        yield from self._load_hf()  

    @abstractmethod
    def _load_local(self, path: Path) -> Iterable[dict]: ...
    @abstractmethod
    def _load_hf(self)        -> Iterable[dict]: ...

    def build_messages(self, sample: dict) -> str:
        messages = [
            {"role": "user", "content": self.prompt_template.format(prompt=sample["prompt"])}
        ]
        return messages

    def check_correctness(self, problem, generation):
        raise NotImplementedError

class GSM8KHandler(DatasetHandler):
    def __init__(self):
        super().__init__()
        self.invalid_ans = "[invalid]"
        self.ans_re = re.compile(r"((-?[$0-9.,]{2,})|(-?[0-9]+))")
        self.gt_re = re.compile(r"#### (\-?[0-9\.\,]+)")
        self.name = "gsm8k"
        self.prompt_template = (
            "Given the following problem, reason and give a final answer to the problem.\nProblem: {prompt}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
        )

    def _load_local(self, path: Path):
        with path.open() as f:
            for line in f:
                j = json.loads(line)
                yield {"id": j.get("id"), "question": j["question"], "prompt": j["question"],
                       "answer": j["answer"].split('####')[-1].strip()}

    def _load_hf(self):
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for i, ex in enumerate(ds):
            yield {"id": i, "question": ex["question"], "prompt": ex["question"],
                   "answer": ex["answer"]}
    def check_correctness(self, problem, generation):
        gt_answer = self.extract_gt_answer(problem["answer"])
        model_answer = extract_answer(generation)
        model_answer = self.sanitize_answer(model_answer)
        return model_answer == gt_answer
    
    def extract_answer(self, generation):
        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return pred
    def extract_gt_answer(self, completion):
        match = self.gt_re.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.invalid_ans

    def sanitize_answer(self, answer):
        patterns_to_remove = [
            ",",  # Remove commas
            r"\$",  # Remove dollar signs
            r"\.$" r"\*",  # Remove trailing period  # Remove asterisks
        ]
        for pattern in patterns_to_remove:
            answer = re.sub(pattern, "", answer)

        matches = self.ans_re.findall(answer)
        if matches:
            # get the last match (i.e final response) and the first / outer capturing group
            match_str = matches[-1][0].strip()
            return match_str
        else:
            print(f'[pred]-----answer: {answer}')
            return self.invalid_ans

class Math500Handler(DatasetHandler):
    name = "math500"
    prompt_template = (
        "Return your final response within \\boxed{{}}. {prompt}"
    )

    def _load_local(self, path: Path):
        with path.open() as f:
            for i, line in enumerate(f):
                j = json.loads(line)
                yield {"id": i, "question": j["question"], "prompt": j["question"],
                       "answer": j["answer"].split('####')[-1].strip()}

    def _load_hf(self):
        ds = load_dataset("qq8933/MATH500", split="test")
        for i, ex in enumerate(ds):
            yield {"id": i, "question": ex["problem"], "prompt": ex["problem"],
                   "answer": ex["answer"].split('####')[-1].strip()}
            
    def check_correctness(self, problem, generation):
        answer = strip_answer_string(problem["answer"])
        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)
    
class AIMETaskHandler(Math500Handler):
    name = "aime25"
    prompt_template = (
        "{prompt}\nReturn your final response within \\boxed{{}}"
    )

    def _load_local(self, path: Path):
        with path.open() as f:
            for i, line in enumerate(f):
                j = json.loads(line)
                yield {"id": i, "question": j["question"], "prompt": j["question"],
                       "answer": j["answer"].split('####')[-1].strip()}
    def _load_hf(self):
        ds_1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        ds_2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        ds   = concatenate_datasets([ds_1, ds_2])   
        for i, ex in enumerate(ds):
            yield {"id": i, "question": ex["question"], "prompt": ex["question"],
                "answer": ex["answer"]}

class GPQADiamondHandler(DatasetHandler):
    name = "gpqa_diamond"
    prompt_template = (
        "Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response. {prompt}" 
    )
    def get_multiple_choice_answers(self, data):
        answers = [
            data["Correct Answer"],
            data["Incorrect Answer 1"],
            data["Incorrect Answer 2"],
            data["Incorrect Answer 3"],
        ]
        random.shuffle(answers)

        options = ["A", "B", "C", "D"]
        options_to_answers = {
            letter: answer for letter, answer in zip(options, answers)
        }

        multiple_choice_string = ", ".join(
            f"{letter}) {options_to_answers[letter]}" for letter in options
        )

        correct_answer_letter = next(
            letter
            for letter, answer in options_to_answers.items()
            if answer == data["Correct Answer"]
        )

        return multiple_choice_string, correct_answer_letter

    def generate_prompt(self, problem):
        multiple_choice_string, correct_answer_letter = (
            self.get_multiple_choice_answers(problem)
        )
        prompt = problem["Question"] + "\n" + multiple_choice_string
        return correct_answer_letter, prompt
    def _load_local(self, path: Path):
        with path.open() as f:
            for idx, line in enumerate(f):
                j = json.loads(line)
                correct_answer_letter, prompt = self.generate_prompt(j)
                yield {"id": idx, "question": j["Question"], "prompt": prompt,
                       "answer": correct_answer_letter}
    def _load_hf(self):
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        for idx, ex in enumerate(ds):
            correct_answer_letter, prompt = self.generate_prompt(ex)
            yield {"id": idx, "question": ex["Question"], "prompt": prompt,
                   "answer": correct_answer_letter}
    def check_correctness(self, problem, generation):
        pred = get_multiple_choice_answer(generation)
        answer = problem["answer"]
        return answer == pred

_HANDLERS = {h.name: h for h in [
                    GSM8KHandler(),
                    Math500Handler(),
                    AIMETaskHandler(),
                    GPQADiamondHandler(),
                ]
            }
def get_dataset_handler(name: str) -> DatasetHandler:
    name = name.lower()
    if name not in _HANDLERS:
        raise KeyError(f"Unknown dataset '{name}'. "
                       f"Known: {list(_HANDLERS.keys())}")
    return _HANDLERS[name]
