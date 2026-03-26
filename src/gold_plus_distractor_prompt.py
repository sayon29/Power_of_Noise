import json
import os

# Configuration
DATASET_DIR = os.path.join("..", "dataset")
PROMPT_DIR = os.path.join("..", "prompts")
INPUT_FILE = os.path.join(DATASET_DIR, "dev.json")

# Parameters
MAX_ANS_TOKENS = 5
REQUIRED_GOLD_DOCS = 2
NUM_DISTRACTORS = 2  # Change this from 0 to 8
LIMIT = 1200  

OUTPUT_FILENAME = f"gold_plus_{NUM_DISTRACTORS}_distractors.jsonl"
OUTPUT_FILE = os.path.join(PROMPT_DIR, OUTPUT_FILENAME)

def tokenize(text):
    return text.strip().split()

def get_docs(sample, n_distractors):
    gold_titles = set(fact[0] for fact in sample["supporting_facts"])
    
    if len(gold_titles) != REQUIRED_GOLD_DOCS:
        return None
        
    gold_docs = []
    distractor_docs = []
    
    for title, sents in sample["context"]:
        doc_text = " ".join(sents)
        if title in gold_titles:
            gold_docs.append(doc_text)
        else:
            distractor_docs.append(doc_text)
    
    # Ensure we have exactly 2 gold docs and enough distractors available
    if len(gold_docs) != REQUIRED_GOLD_DOCS or len(distractor_docs) < n_distractors:
        return None

    #Take the first N distractors in original order
    selected_distractors = distractor_docs[:n_distractors]
    
    # NEAR SETTING : Distractors first, Gold last
    all_docs = selected_distractors + gold_docs
    
    return all_docs

def build_prompt(sample, docs):
    instruction = (
        "Answer the question using ONLY the provided documents. "
        f"Answer in no more than {MAX_ANS_TOKENS} tokens.\n\n"
    )
    
    context_str = ""
    for i, doc in enumerate(docs):
        context_str += f"Document {i+1}: {doc}\n\n"
    
    query = f"Question: {sample['question']}\nAnswer:"
    
    return instruction + context_str + query

try:
    if not os.path.exists(PROMPT_DIR):
        os.makedirs(PROMPT_DIR)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0 

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for sample in data:
            if count >= LIMIT:
                break
                
            ans = sample["answer"]
            if len(tokenize(ans)) > MAX_ANS_TOKENS:
                continue 
            
            docs = get_docs(sample, NUM_DISTRACTORS)
            if docs is None:
                continue

            prompt = build_prompt(sample, docs)
            
            entry = {
                "id": sample["_id"],
                "prompt": prompt,
                "answer": ans
            }
            
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Done! Created '{OUTPUT_FILENAME}'")
    print(f"Total samples made: {count}")

except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}.")