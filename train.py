import argparse, random
import numpy as np
import textattack

from datasets import load_dataset
from textattack.datasets import HuggingFaceDataset
from textattack.training_args import CommandLineTrainingArgs

import nltk
nltk.download('averaged_perceptron_tagger_eng')

def seed_all(s):
    random.seed(s)
    np.random.seed(s)

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--attack", default=None)
p.add_argument("--epochs", type=int, required=True)
p.add_argument("--early-stop", type=int, default=None)

p.add_argument("--n-train", type=int, default=3000)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--eval-batch", type=int, default=32)
p.add_argument("--max-len", type=int, default=128)

p.add_argument("--clean-epochs", type=int, default=1)
p.add_argument("--regen-every", type=int, default=5)
p.add_argument("--adv-examples", type=int, default=-1)
p.add_argument("--query-budget", type=int, default=None)

args = p.parse_args()
seed_all(args.seed)

train = (
    load_dataset("glue", "sst2", split="train")
    .shuffle(seed=args.seed)
    .select(range(args.n_train))
)
valid = load_dataset("glue", "sst2", split="validation")

train_ds = HuggingFaceDataset(train, shuffle=False)
valid_ds = HuggingFaceDataset(valid, shuffle=False)

ta_args = CommandLineTrainingArgs(
    model_name_or_path="lstm",
    dataset="glue^sst2",
    dataset_train_split="train",
    dataset_eval_split="validation",
    attack=args.attack,
    task_type="classification",
    model_num_labels=2,
    model_max_length=args.max_len,

    num_epochs=args.epochs,
    early_stopping_epochs=args.early_stop,
    num_clean_epochs=args.clean_epochs,
    attack_epoch_interval=args.regen_every,
    num_train_adv_examples=args.adv_examples,
    query_budget_train=args.query_budget,

    learning_rate=args.lr,
    num_warmup_steps=0,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.eval_batch,
    random_seed=args.seed,

    output_dir=args.out,
    save_last=True,
    load_best_model_at_end=False,
)

seed_all(args.seed)

model = CommandLineTrainingArgs._create_model_from_args(ta_args)
attack = CommandLineTrainingArgs._create_attack_from_args(ta_args, model)

trainer = textattack.Trainer(
    model,
    task_type="classification",
    attack=attack,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    training_args=ta_args,
)

trainer.train()
