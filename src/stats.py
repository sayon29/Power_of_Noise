import json
import os
import math

DATASET_DIR = os.path.join("..", "dataset")
FILE = os.path.join(DATASET_DIR, "dev.json")

def tokenize(text):
    return text.strip().split()

try:
    with open(FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    max_len = 0
    longest_answers = []
    lengths = []

    for sample in data:
        ans = sample["answer"]
        tokens = tokenize(ans)
        l = len(tokens)
        lengths.append(l)
        
        if l > max_len:
            max_len = l
            longest_answers = [ans]
        elif l == max_len:
            if ans not in longest_answers:
                longest_answers.append(ans)

    n = len(lengths)
    if n > 0:
        mean = sum(lengths) / n
        variance = sum((x - mean) ** 2 for x in lengths) / n
        std_dev = math.sqrt(variance)

        print(f"Max ans token length: {max_len}")
        print(f"Average: {mean:.2f}")
        print(f"Variance: {variance:.2f}")
        print(f"Std Dev: {std_dev:.2f}")
        print("\nExamples:")
        for a in longest_answers[:10]:
            print("-", a)

except FileNotFoundError:
    print(f"Error: {FILE} not found.")