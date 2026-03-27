import json
import os
import math

PROMPTS_DIR = os.path.join("..", "prompts")
DATASET_DIR = os.path.join("..", "dataset")
FILE = os.path.join(DATASET_DIR, "dev.json")

def tokenize(text):
    return text.strip().split()

try:
    with open(FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    ans_lengths = []
    gold_doc_counts = []
    total_doc_counts = []
    
    max_ans_len = 0
    longest_answers = []

    for sample in data:
        ans = sample["answer"]
        tokens = tokenize(ans)
        l = len(tokens)
        ans_lengths.append(l)
        
        if l > max_ans_len:
            max_ans_len = l
            longest_answers = [ans]
        elif l == max_ans_len:
            if ans not in longest_answers:
                longest_answers.append(ans)

        gold_titles = set(fact[0] for fact in sample["supporting_facts"])
        gold_doc_counts.append(len(gold_titles))
        
        total_doc_counts.append(len(sample["context"]))

    n = len(data)
    if n > 0:
        def get_stats(data_list):
            mu = sum(data_list) / n
            var = sum((x - mu) ** 2 for x in data_list) / n
            return mu, var, math.sqrt(var)

        ans_mu, ans_var, ans_std = get_stats(ans_lengths)
        gold_mu, gold_var, gold_std = get_stats(gold_doc_counts)
        total_mu, total_var, total_std = get_stats(total_doc_counts)

        print(f"--- Dataset Statistics ({n} samples) ---")
        print(f"Answer Tokens:")
        print(f"  Min: {min(ans_lengths)} | Max: {max_ans_len} | Avg: {ans_mu:.2f} (Std: {ans_std:.2f})")
        print("-" * 30)
        print(f"Gold Docs per Question:")
        print(f"  Min: {min(gold_doc_counts)} | Max: {max(gold_doc_counts)} | Avg: {gold_mu:.2f} (Std: {gold_std:.2f})")
        print("-" * 30)
        print(f"Total Docs per Question:")
        print(f"  Min: {min(total_doc_counts)} | Max: {max(total_doc_counts)} | Avg: {total_mu:.2f} (Std: {total_std:.2f})")
        print("\nLongest Answer Examples:")
        for a in longest_answers[:5]:
            print("-", a)

except FileNotFoundError:
    print(f"Error: {FILE} not found.")