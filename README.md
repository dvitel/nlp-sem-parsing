# Code Synthesis and semantic parsing with DistilGPT2 

Code repo: [Github Repo](https://github.com/dvitel/nlp-sem-parsing). \
Files g0-gX.py contain experiments with GEO dataset. \
Files h0-hX.py - experiments with Hearthstone dataset.
Preprocessing of gX and hX files is explained in the presentation. 

Datasets repo:
1. [GEO dataset](https://huggingface.co/datasets/dvitel/geo)
2. [Hearthstone dataset](https://huggingface.co/datasets/dvitel/hearthstone)

Metrics repo:
1. [CodeBLEU](https://huggingface.co/spaces/dvitel/codebleu)

Models repo:
1. [h0 fine-tuned model: distilgpt2 on hearthstone baseline](https://huggingface.co/dvitel/h0)
2. [h0-1 fine-tuned model: CodeGPT-small-py on hearthstone baseline](https://huggingface.co/dvitel/h0-1)
3. [h1 fine-tuned model: AST dump representation](https://huggingface.co/dvitel/h1)
4. [h3 fine-tuned model: Local name removal](https://huggingface.co/dvitel/h3)
