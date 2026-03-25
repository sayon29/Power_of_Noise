import json
import os

DATASET_DIR = os.path.join("..", "dataset")
PROMPT_DIR = os.path.join("..", "prompts")
INPUT_FILE = os.path.join("..", "dataset", "dev.json")
OUTPUT_FILE = os.path.join(PROMPT_DIR, "gold_prompts_wiki.jsonl")

MAX_ANS_TOKENS = 5

def tokenize(text):
    return text.strip().split()

def get_gold_docs(sample):
    gold_titles = set(fact[0] for fact in sample["supporting_facts"])
    
    docs = []
    for title, sents in sample["context"]:
        if title in gold_titles:
            docs.append(" ".join(sents))
    
    return docs

def build_prompt(sample, docs):
    instruction = (
        "Answer the question using ONLY the provided documents. "
        f"Answer in no more than {MAX_ANS_TOKENS} tokens. "
        "If the answer is not present, output NO-RES.\n\n"
    )
    
    context_str = ""
    for i, doc in enumerate(docs):
        context_str += f"Document {i+1}: {doc}\n\n"
    
    query = f"Question: {sample['question']}\nAnswer:"
    
    return instruction + context_str + query

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0 

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for sample in data:
            ans = sample["answer"]
            
            # Skip if answer is too long
            if len(tokenize(ans)) > MAX_ANS_TOKENS:
                continue 
            
            docs = get_gold_docs(sample)
            
            if not docs:
                continue

            prompt = build_prompt(sample, docs)
            
            entry = {
                "id": sample["_id"],
                "prompt": prompt,
                "answer": ans
            }
            
            out.write(json.dumps(entry) + "\n")
            count += 1

    print(f"Done! Processed {len(data)} samples.")
    print(f"Gold prompts generated: {count}")

except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}. Check if the 'dataset' folder is in the parent directory.")