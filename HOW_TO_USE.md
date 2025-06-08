## How to use

medqa_thinking_env.py

terminal 1:
```
run-api
```

terminal 2:
```
python environments/medqa_thinking_env.py serve --slurm false 
```

terminal 3:
```
TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --port 9001
```

terminal 4:
```
view-run --port 9002 --tokenizer Qwen/Qwen2.5-1.5B-Instruct
```