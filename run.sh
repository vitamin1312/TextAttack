#!/usr/bin/env bash
set -euo pipefail

export PYTHONWARNINGS="ignore::SyntaxWarning"

rm -rf outputs results
mkdir -p outputs results

SEED=42
N_TRAIN=3000
N_EVAL=872

COMMON=(
  --n-train "$N_TRAIN"
  --seed "$SEED"
  --lr 3e-4
  --batch 32
  --eval-batch 32
  --max-len 128
)

python train.py "${COMMON[@]}" \
  --out outputs/baseline \
  --epochs 75 \
  --early-stop 5

python train.py "${COMMON[@]}" \
  --out outputs/deepwordbug20 \
  --attack deepwordbug \
  --epochs 20 \
  --clean-epochs 1 \
  --regen-every 1 \
  --adv-examples -1

python train.py "${COMMON[@]}" \
  --out outputs/deepwordbug75 \
  --attack deepwordbug \
  --epochs 75 \
  --clean-epochs 1 \
  --regen-every 1 \
  --adv-examples -1

python train.py "${COMMON[@]}" \
  --out outputs/textfooler20 \
  --attack textfooler \
  --epochs 20 \
  --clean-epochs 1 \
  --regen-every 1 \
  --adv-examples -1

for dir in outputs/*; do
  run="$(basename "$dir")"

  if [[ "$run" == "baseline" ]]; then
    model="$dir/best_model"
  else
    model="$dir/last_model"
  fi

  for recipe in deepwordbug textfooler pruthi hotflip bae; do
    python attack.py \
      --model "$model" \
      --recipe "$recipe" \
      --n "$N_EVAL" \
      --seed "$SEED" \
      --json "results/${run}_under_${recipe}.json" \
      --csv "results/${run}_under_${recipe}.csv"
  done
done
