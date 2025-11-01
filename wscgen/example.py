from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from wscgen.chopper import Chopper
from wscgen.generate import wsc_generate
from wscgen.prober import build_prober
from wscgen.utils import find_newline_token_ids, set_seed

set_seed(41)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the trained prober
prober_file = hf_hub_download(
    repo_id="xiewenya/DeepSeek-R1-Distilled-Qwen-7B-s1-classifier",
    filename="probe.pkl",
    repo_type="model"
)

prober_file = hf_hub_download(
    repo_id="xiewenya/WordSaladChopper_Classifier",
    filename="DeepSeek-R1-Distill-Qwen-7B_s1/probe.pkl",
    repo_type="model",
)

prober = build_prober("logistic").load(prober_file)
# Initialize chopper
chopper = Chopper(
    tokenizer=tokenizer, detector=prober,
    thresh=0.5, streak_len=2, short_streak_len=5, len_threshold=10
)

messages = [
    {"role": "user", "content": "Repeat the following sentence 10 times with two newlines between each repetition and without any other text. The sentence is: Today is Halloween, and I'm going to hang out with my friends, what a great day!"}
]
prompt_txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

newline_token_ids = find_newline_token_ids(tokenizer)
gen_cfg = {"temperature": 0.6, "top_p": 0.95}


result_wsc = wsc_generate(
    model, tokenizer, prompt_txt, chopper,
    newline_token_ids=newline_token_ids, gen_cfg=gen_cfg,
    rescue_prompt="I can find a clearer solution if I focus on the core problem.",
    token_budget=32768, rescue_budget=128, max_rescues=1
)

print("Generated text:", result_wsc["response"])
print("Total tokens used:", result_wsc["total_used_tokens"])