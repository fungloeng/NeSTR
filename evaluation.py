import os
import re
import json
import time
import string
import random
import argparse
import unicodedata
import numpy as np
from tqdm import tqdm
from openai import OpenAI

# ========================
# Argument parsing
# ========================
parser = argparse.ArgumentParser(description="Evaluate LLM predictions using EM and F1.")
parser.add_argument('--model_name', type=str, default="gpt-4o-mini-2024-07-18", help="API model name")
parser.add_argument('--dataset', type=str, required=True, help="Path to input JSONL file")
parser.add_argument('--sample', type=int, default=0, help="Optional: number of examples to subsample")
args = parser.parse_args()

# ========================
# File and model setup
# ========================
model_name = args.model_name
data_path = args.dataset
sample_size = args.sample or None
max_new_tokens = 1024

# Auto-generate output file name
filename_no_ext = os.path.splitext(os.path.basename(data_path))[0]
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_tag = model_name.replace("/", "_")
output_file = f"results/api/{model_tag}_{filename_no_ext}_{sample_size or 'full'}_{timestamp}.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# ========================
# API client configuration
# ========================

api_key = os.getenv("YOUR_API_KEY")

client = OpenAI(api_key=api_key, base_url=base_url)

# ========================
# Utility functions
# ========================
def extract_answer(text):
    """Extract text inside <answer>...</answer> tags."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else "Unknown"

def decode_unicode_escapes(text):
    return re.sub(r'u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)

def normalize_answer(text):
    """Standardize text for comparison: normalize, strip punctuation/articles, uppercase."""
    text = decode_unicode_escapes(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.replace('_', ' ')
    text = re.sub(r'\b(a|an|the)\b', ' ', text, flags=re.IGNORECASE)
    text = ''.join(c for c in text if c not in string.punctuation)
    return ' '.join(text.upper().split())

def exact_match(pred, gold):
    return int(pred.strip() == gold.strip())

def f1_score(pred, gold):
    """Compute token-level F1."""
    pred_tokens = pred.strip().split()
    gold_tokens = gold.strip().split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# ========================
# Call model API to generate predictions
# ========================
def generate_prediction(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model_name,
                input=prompt,
                temperature=0.1,
                max_output_tokens=max_new_tokens
            )
            prediction = response.output_text.strip()
            extracted = extract_answer(prediction)
            usage = response.usage
            return (
                prediction,
                extracted,
                usage.input_tokens if usage else 0,
                usage.output_tokens if usage else 0
            )
        except Exception as e:
            print(f"[Retry {attempt+1}] Error: {e}")
    return "", "Unknown", 0, 0

# ========================
# Generate and save predictions
# ========================
def generate_and_save_predictions(input_path, output_path, sample_n=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    if sample_n and len(records) > sample_n:
        records = random.sample(records, sample_n)

    results = []
    print(f"Evaluating {len(records)} samples...")
    for item in tqdm(records, desc="Generating"):
        prompt = item.get("prompt")
        gold_answer = item.get("answer")
        if not prompt or not gold_answer:
            continue
        response, extracted, in_tok, out_tok = generate_prediction(prompt)
        results.append({
            "question": item.get("question"),
            "answer": gold_answer,
            "prompt": prompt,
            "model_response": response,
            "extracted_answer": extracted,
            "input_tokens": in_tok,
            "output_tokens": out_tok
        })

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for r in results:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"→ Predictions saved to {output_path}")

# ========================
# Evaluate predictions using EM and F1
# ========================
def evaluate_predictions(prediction_path):
    with open(prediction_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f if line.strip()]

    ems, f1s = [], []
    for entry in tqdm(lines, desc="Evaluating"):
        pred, gold = entry.get("extracted_answer", ""), entry.get("answer", "")
        norm_pred = normalize_answer(pred)
        norm_gold = normalize_answer(gold)
        em = exact_match(norm_pred, norm_gold)
        f1 = f1_score(norm_pred, norm_gold)
        entry.update({"em": em, "f1": f1})
        ems.append(em)
        f1s.append(f1)

    avg_em = np.mean(ems) if ems else 0.0
    avg_f1 = np.mean(f1s) if f1s else 0.0
    summary = {
        "type": "average",
        "total": len(ems),
        "em": avg_em,
        "f1": avg_f1
    }
    lines.append(summary)

    with open(prediction_path, 'w', encoding='utf-8') as f_out:
        for item in lines:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n=== Evaluation Summary ===")
    print(f"Samples Evaluated: {summary['total']}")
    print(f"Exact Match (EM): {avg_em * 100:.2f}%")
    print(f"F1 Score: {avg_f1 * 100:.2f}%")

# ========================
# Entry point
# ========================
if __name__ == "__main__":
    generate_and_save_predictions(data_path, output_file, sample_size)
    evaluate_predictions(output_file)
