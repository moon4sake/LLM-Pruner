## Supported models
- [x] [Llama-3.1]
- [x] [Qwen-2.5]

## Prune and fine-tune
### All at once: 
```
bash scripts/run.sh
```
### Prune
```
python prune.py
```

### Fine-tune
```
python post_training.py
```

## Evaluate
### Latency
```
bash eval/latency/test_latency.sh
```

### S1 (simple scaling)
```
bash eval/s1/run.sh
```

### GSM8k
```
bash eval/gsm8k/test_cot.sh
```
