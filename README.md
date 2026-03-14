# Text2Code

Natural language → simple expression language (arithmetic, conditionals, lambdas).

Seq2Seq transformer architecture

## Checkpoints

- **checkpoint.pt** — Saved every 5 epochs
- **checkpoint_best.pt** — Best validation accuracy (use for inference)

## Training

```bash
python data_generator.py -n 100000 -d 4 -o data_large --split --seed 42
python train.py --data data_large_train.jsonl --model-size large --save checkpoint.pt --batch-size 256
```

## Inference

```bash
python infer.py --checkpoint checkpoint_best.pt
```
