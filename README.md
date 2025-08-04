# NeSTR (Anonymous Submission)

This repository contains code and data for our anonymous AAAI submission.

## Files

- `evaluation.py`: run model prediction and compute EM/F1 scores
- `Information_Flow/attention_flow.py`: extract attention-based information flow
- `Information_Flow/plot_attention_flows.py`: visualize information flow
- `data/build_prompt.py`: construct prompts from JSON input
- `data/context_test.json`: example input data
- `Information_Flow/prompt_GOLD.json`: example prompt input

## How to Run

Generate predictions:

```
python evaluation.py --model_name gpt-4o --dataset data/context_test.json
```

Analyze information flow:

```
python Information_Flow/attention_flow.py \
  --mode multiple --i 2 --f_name example --path Information_Flow/prompt_GOLD.json
```

Plot attention flow:

```
python Information_Flow/plot_attention_flows.py --f_name example --flows ca cq
```

## Notes

- This repository is anonymized for blind review.
