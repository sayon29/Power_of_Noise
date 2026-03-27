import json
import os
import random

# Global Constants
DATASET_DIR = os.path.join("..", "dataset")
PROMPT_DIR = os.path.join("..", "prompts")
INPUT_FILE = os.path.join(DATASET_DIR, "dev.json")
POOL_FILE = os.path.join(DATASET_DIR, "random_doc_pool.json")

# Fixed parameters
MAX_ANS_TOKENS = 5
REQUIRED_GOLD_DOCS = 2
LIMIT = 1200

def tokenize(text):
    return text.strip().split()

def get_docs(sample, n_distractors, n_random, global_pool):
    gold_titles = set(fact[0] for fact in sample["supporting_facts"])
    if len(gold_titles) != REQUIRED_GOLD_DOCS:
        return None
        
    gold_docs, distractor_docs = [], []
    for title, sents in sample["context"]:
        doc_text = " ".join(sents)
        if title in gold_titles:
            gold_docs.append(doc_text)
        else:
            distractor_docs.append(doc_text)
    
    if len(gold_docs) != REQUIRED_GOLD_DOCS or len(distractor_docs) < n_distractors:
        return None

    selected_distractors = distractor_docs[:n_distractors]
    
    # Filter pool to avoid duplicates with gold docs
    potential_randoms = [d for d in global_pool if d not in gold_docs]
    if len(potential_randoms) < n_random:
        return None
    
    selected_randoms = random.sample(potential_randoms, n_random)
    return selected_randoms + selected_distractors + gold_docs

def build_prompt(sample, docs):
    instruction = f"Answer the question using ONLY the provided documents. Answer in no more than {MAX_ANS_TOKENS} tokens.\n\n"
    context_str = "".join([f"Document {i+1}: {doc}\n\n" for i, doc in enumerate(docs)])
    return instruction + context_str + f"Question: {sample['question']}\nAnswer:"

def generate_dataset(num_dist, num_rand):
    """The function your master script will call"""
    output_filename = f"gold_2_dist_{num_dist}_rand_{num_rand}.jsonl"
    output_path = os.path.join(PROMPT_DIR, output_filename)

    if not os.path.exists(PROMPT_DIR):
        os.makedirs(PROMPT_DIR)

    with open(POOL_FILE, "r", encoding="utf-8") as f:
        global_pool = json.load(f)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0 
    with open(output_path, "w", encoding="utf-8") as out:
        for sample in data:
            if count >= LIMIT: break
            if len(tokenize(sample["answer"])) > MAX_ANS_TOKENS: continue 
            
            docs = get_docs(sample, num_dist, num_rand, global_pool)
            if docs is None: continue

            entry = {"id": sample["_id"], "prompt": build_prompt(sample, docs), "answer": sample["answer"]}
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Created '{output_filename}' with {count} samples.")