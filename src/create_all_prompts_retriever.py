from prompt_gen_retriever import generate_bm25_dataset

def run_bm25_suite():
    FIXED_K = 4
    RANDOM_VALUES = [0, 1, 2, 4, 6, 8, 10]

    print("--- Starting BM25 Noise Experiments ---")
    for r in RANDOM_VALUES:
        print(f"Generating for: k={FIXED_K}, random_noise={r}")
        generate_bm25_dataset(FIXED_K, r)

if __name__ == "__main__":
    run_bm25_suite()
    print("\nAll BM25 prompt files have been created in the prompts directory.")