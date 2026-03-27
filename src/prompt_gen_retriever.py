import json
import os
import random
from bm25 import BM25Retriever

# Configuration
DATASET_DIR = os.path.join("..", "dataset")
PROMPT_DIR = os.path.join("..", "prompts")
POOL_FILE = os.path.join(DATASET_DIR, "random_doc_pool.json")
INPUT_FILE = os.path.join(DATASET_DIR, "dev.json")

# Parameters
LIMIT = 1200 
MAX_ANS_TOKENS = 5

def generate_bm25_dataset(num_k, num_rand):
    if not os.path.exists(PROMPT_DIR):
        os.makedirs(PROMPT_DIR)

    with open(POOL_FILE, "r", encoding="utf-8") as f:
        global_pool = json.load(f)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize BM25 with the pool
    retriever = BM25Retriever(global_pool)

    output_filename = f"bm25_k{num_k}_rand{num_rand}.jsonl"
    output_path = os.path.join(PROMPT_DIR, output_filename)
    
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for sample in data:
            if count >= LIMIT:
                break
            
            ans = sample["answer"]
            if len(ans.strip().split()) > MAX_ANS_TOKENS:
                continue

            # Retrieve top K documents
            retrieved_docs = retriever.retrieve(sample["question"], num_k)
            
            # Sample N random docs (excluding retrieved docs)
            potential_randoms = [d for d in global_pool if d not in retrieved_docs]
            if len(potential_randoms) < num_rand:
                continue
            selected_randoms = random.sample(potential_randoms, num_rand)
            
            # Order: Random Noise first, Retrieved Docs last (Near Query)
            all_docs = selected_randoms + retrieved_docs
            
            # Build Prompt
            prompt = f"Answer the question using ONLY the provided documents. Answer in no more than {MAX_ANS_TOKENS} tokens.\n\n"
            for i, doc in enumerate(all_docs):
                prompt += f"Document {i+1}: {doc}\n\n"
            
            prompt += f"Question: {sample['question']}\nAnswer:"

            entry = {"id": sample["_id"], "prompt": prompt, "answer": ans}
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"Finished: {output_filename} ({count} samples)")