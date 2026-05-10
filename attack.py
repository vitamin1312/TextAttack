import argparse

import textattack
from datasets import load_dataset
from textattack.datasets import HuggingFaceDataset
from textattack.models.helpers import LSTMForClassification
from textattack.models.wrappers import PyTorchModelWrapper
from textattack.attack_recipes import (
    DeepWordBugGao2018,
    TextFoolerJin2019,
    Pruthi2019,
    HotFlipEbrahimi2017,
    BAEGarg2019,
)

import nltk
nltk.download('averaged_perceptron_tagger_eng')


RECIPES = {
    "deepwordbug": DeepWordBugGao2018,
    "textfooler": TextFoolerJin2019,
    "pruthi": Pruthi2019,
    "hotflip": HotFlipEbrahimi2017,
    "bae": BAEGarg2019,
}


p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--recipe", required=True, choices=RECIPES)
p.add_argument("--json", required=True)
p.add_argument("--csv", required=True)
p.add_argument("--n", type=int, default=872)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

model = LSTMForClassification.from_pretrained(args.model)
wrapper = PyTorchModelWrapper(model, model.tokenizer)

attack = RECIPES[args.recipe].build(wrapper)

dataset = HuggingFaceDataset(
    load_dataset("glue", "sst2", split="validation"),
    shuffle=False,
)

attack_args = textattack.AttackArgs(
    num_examples=args.n,
    random_seed=args.seed,
    log_summary_to_json=args.json,
    log_to_csv=args.csv,
    disable_stdout=True,
)

textattack.Attacker(attack, dataset, attack_args).attack_dataset()
