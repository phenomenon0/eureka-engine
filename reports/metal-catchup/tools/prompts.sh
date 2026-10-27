#!/bin/bash
# the test prompts as ids (llama.cpp tokenizer of the 0.8B GGUF; the Qwen3.5 vocabulary is shared)
TOK=~/llama.cpp/build/bin/llama-tokenize; . ./paths.sh; G=$M08
tok() { $TOK -m $G -p "$1" --ids --log-disable 2>/dev/null | tail -1 | tr -d "[],"; }
P1=$(tok "The capital of France is")
P2=$(tok "Write a short story about a robot who learns to paint. Once upon a time,")
P3=$(tok "$(printf "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n")")
