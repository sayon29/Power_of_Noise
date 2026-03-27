import json
import os

DATASET_DIR = os.path.join("..", "dataset")
INPUT_FILE = os.path.join(DATASET_DIR, "dev.json")
POOL_FILE = os.path.join(DATASET_DIR, "random_doc_pool.json")

def create_pool():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use a dict to keep track of unique documents by title
    unique_docs = {}
    for sample in data:
        for title, sents in sample["context"]:
            if title not in unique_docs:
                unique_docs[title] = " ".join(sents)

    # Save only the text of the documents as a list
    pool = list(unique_docs.values())
    
    with open(POOL_FILE, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=4)
    
    print(f"Pool created with {len(pool)} unique documents at {POOL_FILE}")

if __name__ == "__main__":
    create_pool()