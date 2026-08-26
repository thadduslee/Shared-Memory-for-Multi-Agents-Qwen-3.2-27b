# Medical split - full results

| Model | Memory agent | Judge | Utility % | Privacy leak % | Deletion leak % | MGS % |
|---|---|---|---:|---:|---:|---:|
| Qwen3.8-27B | Long-Context | GPT-4.1 | 90.95 | 4.69 | 0.00 | 86.69 |
| Qwen3.8-27B | RAG (naive) | GPT-4.1 | 65.24 | 10.94 | 2.82 | 56.46 |
| Qwen3.8-27B | RAG (policy) | GPT-4.1 | 39.05 | 5.21 | 2.26 | 36.18 |
| Qwen3.8-27B | A-Mem | GPT-4.1 | 62.38 | 11.98 | 3.39 | 53.05 |
| Qwen3.8-27B | Mem0 | GPT-4.1 | 45.24 | 14.06 | 1.69 | 38.22 |
| Qwen3.8-27B | ReMeM (iterative) | GPT-4.1 | 55.24 | 13.54 | 5.65 | 45.06 |
| Qwen3.8-27B | ReMeM (single) | GPT-4.1 | 51.90 | 14.58 | 6.21 | 41.58 |
| Qwen3.8-27B | Long-Context | rule-based | 43.81 | 10.94 | 0.56 | 38.80 |
| Qwen3.8-27B | RAG (naive) | rule-based | 29.52 | 16.67 | 5.08 | 23.35 |
| Qwen3.8-27B | RAG (policy) | rule-based | 16.19 | 11.46 | 6.78 | 13.36 |
| Qwen3.8-27B | A-Mem | rule-based | 30.95 | 16.15 | 3.39 | 25.08 |
| Qwen3.8-27B | Mem0 | rule-based | 18.10 | 22.40 | 2.82 | 13.65 |
| Qwen3.8-27B | ReMeM (iterative) | rule-based | 29.52 | 15.10 | 6.78 | 23.37 |
| Qwen3.8-27B | ReMeM (single) | rule-based | 26.67 | 16.67 | 7.34 | 20.59 |
| Qwen2.5-32B-Instruct | Long-Context | GPT-4.1 | 87.14 | 23.44 | 14.12 | 57.30 |
| Qwen2.5-32B-Instruct | RAG (naive) | GPT-4.1 | 67.14 | 44.27 | 27.12 | 27.27 |
| Qwen2.5-32B-Instruct | RAG (policy) | GPT-4.1 | 36.67 | 19.27 | 10.17 | 26.59 |
| Qwen2.5-32B-Instruct | A-Mem | GPT-4.1 | 65.24 | 46.88 | 27.12 | 25.26 |
| Qwen2.5-32B-Instruct | Mem0 | GPT-4.1 | 52.38 | 45.31 | 30.51 | 19.91 |
| Qwen2.5-32B-Instruct | ReMeM (iterative) | GPT-4.1 | 50.95 | 42.71 | 29.38 | 20.62 |
| Qwen2.5-32B-Instruct | ReMeM (single) | GPT-4.1 | 45.71 | 38.02 | 23.73 | 21.61 |
