import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================================
# Analyze attention-based flows
# ================================
def analyze_information_flows(layer_interval, prompt_template, question, contexts, answer, prompt_name, prompt_org):
    model_path = "path/to/your/local/Qwen-model"  # replace with actual local model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Format input with chat template
    messages = [{"role": "user", "content": prompt_org}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    raw_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    encoding = tokenizer(text, return_offsets_mapping=True)
    offset_mapping = encoding["offset_mapping"] + [(len(text), len(text) + 1)]

    def find_token_range(part_text):
        start = text.find(part_text)
        if start == -1:
            raise ValueError(f"Text part not found: {part_text}")
        end = start + len(part_text)
        start_idx = next(i for i, (s, e) in enumerate(offset_mapping) if s <= start < e)
        try:
            end_idx = next(i for i, (s, e) in enumerate(offset_mapping) if s <= end <= e)
        except StopIteration:
            end_idx = len(offset_mapping) - 1
        return (start_idx, end_idx)

    part_indices = {
        "prompt_template": find_token_range(prompt_template),
        "contexts": find_token_range(contexts),
        "question": find_token_range(question),
        "answer": find_token_range(answer)
    }

    with torch.no_grad():
        outputs = model(**raw_inputs, output_attentions=True)
    attentions = outputs.attentions
    if attentions is None:
        raise ValueError("No attention matrices returned")

    total_layers = len(attentions)
    selected_layers = list(range(total_layers)[::layer_interval])

    flows = {k: [] for k in [
        "ca", "ac", "qc", "cq", "cp", "pc", "aq", "qa", "ap", "pa", "qp", "pq"
    ]}

    def compute_flow(attn, from_range, to_range):
        from_idx = torch.arange(from_range[0], from_range[1] + 1)
        to_idx = torch.arange(to_range[0], to_range[1] + 1)
        mask = (from_idx[:, None] != to_idx[None, :]).to(attn.device)
        submatrix = attn[:, from_range[0]:from_range[1] + 1, to_range[0]:to_range[1] + 1] * mask
        return submatrix.sum().item()

    for l in selected_layers:
        attn_layer = attentions[l].squeeze(0)
        p_start, p_end = part_indices["prompt_template"]
        c_start, c_end = part_indices["contexts"]
        q_start, q_end = part_indices["question"]
        a_start, a_end = part_indices["answer"]
        flows["ca"].append(compute_flow(attn_layer, (c_start, c_end), (a_start, a_end)))
        flows["ac"].append(compute_flow(attn_layer, (a_start, a_end), (c_start, c_end)))
        flows["qc"].append(compute_flow(attn_layer, (q_start, q_end), (c_start, c_end)))
        flows["cq"].append(compute_flow(attn_layer, (c_start, c_end), (q_start, q_end)))
        flows["cp"].append(compute_flow(attn_layer, (c_start, c_end), (p_start, p_end)))
        flows["pc"].append(compute_flow(attn_layer, (p_start, p_end), (c_start, c_end)))
        flows["aq"].append(compute_flow(attn_layer, (a_start, a_end), (q_start, q_end)))
        flows["qa"].append(compute_flow(attn_layer, (q_start, q_end), (a_start, a_end)))
        flows["ap"].append(compute_flow(attn_layer, (a_start, a_end), (p_start, p_end)))
        flows["pa"].append(compute_flow(attn_layer, (p_start, p_end), (a_start, a_end)))
        flows["qp"].append(compute_flow(attn_layer, (q_start, q_end), (p_start, p_end)))
        flows["pq"].append(compute_flow(attn_layer, (p_start, p_end), (q_start, q_end)))

    # Generation for thinking + answer
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        split_index = len(output_ids) - output_ids[::-1].index(151668)  # Replace with </answer> token ID if needed
    except ValueError:
        split_index = 0

    return {
        "prompt_name": prompt_name,
        "prompt_template": prompt_template,
        "layer_interval": layer_interval,
        "selected_layers": selected_layers,
        "flows": flows,
        "thinking_content": tokenizer.decode(output_ids[:split_index], skip_special_tokens=True).strip(),
        "generated_text": tokenizer.decode(output_ids[split_index:], skip_special_tokens=True).strip()
    }

# Save result to JSON
def save_experiment_data(data, output_dir="results/data", file_suffix=""):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{file_suffix}_{data['prompt_name']}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filename}")
    return filename

# Run on multiple prompt examples
def run_multiple_experiments_and_save(args):
    with open(args.path, 'r', encoding="utf-8") as f:
        prompt_list = [json.loads(line) for line in f if line.strip()]

    for prompt in tqdm(prompt_list, desc="Running Experiments"):
        result = analyze_information_flows(
            layer_interval=args.i,
            prompt_template=prompt["prompt_template"],
            question=prompt["question"],
            contexts=prompt["contexts"],
            answer=prompt["answer"],
            prompt_name=prompt["prompt_name"],
            prompt_org=prompt["prompt_org"]
        )
        save_experiment_data(result, file_suffix=args.f_name)
        print(f"→ [Thinking]: {result['thinking_content']}\n→ [Answer]: {result['generated_text']}\n")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['multiple'], default='multiple', help="Execution mode")
    parser.add_argument('--i', type=int, default=1, help="Layer interval for attention sampling")
    parser.add_argument('--f_name', type=str, required=True, help="Suffix for output filenames")
    parser.add_argument('--path', type=str, required=True, help="Path to input JSONL prompt file")
    args = parser.parse_args()

    if args.mode == 'multiple':
        run_multiple_experiments_and_save(args)
    else:
        raise NotImplementedError("Only 'multiple' mode is implemented.")
