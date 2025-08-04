import json
import argparse

# ------------------------------
# Argument parser
# ------------------------------
parser = argparse.ArgumentParser(description="Add refnesy prompt to a JSONL file.")
parser.add_argument('--input', type=str, required=True, help="Path to input JSONL file")
parser.add_argument('--output', type=str, required=True, help="Path to save output JSONL file")
args = parser.parse_args()

input_file = args.input
output_file = args.output

# ------------------------------
# Neuro-Symbolic Prompt Template (refnesy)
# ------------------------------
PROMPT_TEMPLATE = """### Question:\n\nYou are an AI assistant that uses a Neuro-Symbolic Reasoning approach to answer queries. Follow the structured steps below. You must treat entities and events symbolically and apply logical inference rules grounded in the provided temporal knowledge.\n\nStep 1. Symbolically represent all relevant facts and relations from the temporal context using predicate-style expressions (e.g., relation(subject, object, start time, end time)).\nStep 2. Apply inference rules and symbolic matching inside the <inference> tags to derive the answer using the temporal context. Use the symbolic representations defined above.\nStep 3. Evaluate the correctness and consistency of your logic in the <consistency_check> tags.\nStep 4. If inconsistencies or gaps are found, reflect in the <reflection> section: revise symbolic representations or inference steps based on logical necessity or domain knowledge.\nStep 5. Output your final, concise answer within the <answer> tags. If the answer is a number, just write the number. Otherwise, write only the name of the team, event, or entity.\n\nImportant: All symbolic reasoning, representations, and logical steps must be contained within the thinking section. Ensure that all entity names preserve original spelling, accents, and encodings. Only place the result in the answer section.\n\nUse the following format:\n\n<symbolic_representation>\n[Define all relevant symbolic relationships derived from context.]\n</symbolic_representation>\n\n<inference>\n[Apply logical rules to infer the answer from the symbolic representation.]\n</inference>\n\n<consistency_check>\n[Verify consistency and completeness of inference steps.]\n</consistency_check>\n\n<reflection>\n[If needed, revise symbolic representation or inference steps. Justify what is changed and why.]\n</reflection>\n\n<answer>\n[Your final answer.]\n</answer>\n\nQuestion: {question}\n\nTemporal context: {temporal_context}\n\n### Answer:"""

# ------------------------------
# Prompt Construction Function
# ------------------------------
def add_prompt(entry):
    question = entry.get("question", "").strip()
    temporal = entry.get("temporal_context", "").strip()
    prompt = PROMPT_TEMPLATE.format(question=question, temporal_context=temporal)
    entry["prompt"] = prompt
    return entry

# ------------------------------
# File Processing
# ------------------------------
def process_file_add_prompt(input_path, output_path):
    enriched = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                obj = json.loads(line)
                enriched.append(add_prompt(obj))
            except json.JSONDecodeError:
                continue

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in enriched:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

# ------------------------------
# Run Script
# ------------------------------
process_file_add_prompt(input_file, output_file)
