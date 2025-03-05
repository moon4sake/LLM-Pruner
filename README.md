## Supported models
- [x] [Llama-3.1, 3.2]
- [x] [Qwen-2.5]
- [x] [Deepseek-R1-Distill models]

## Prune and fine-tune
### All at once: 
```
bash examples/run.sh
```
### Prune
```
python scripts/prune_v1.py
```

### Fine-tune
```
python python/finetune_v1.py
```

## Evaluate
### Latency
```
bash scripts/eval/latency/test_latency.sh
```

### S1 (simple scaling)
```
bash scripts/eval/s1/run.sh
```

### GSM8k
```
bash scripts/eval/gsm8k/run.sh
```
