from prompt_gen_gold import generate_dataset

def run_experiments():
    # List of (num_distractors, num_random)
    variations = [
        (0, 10), (0, 12)
    ]

    for d, r in variations:
        print(f"Running config: Distractors={d}, Random={r}...")
        generate_dataset(num_dist=d, num_rand=r)

if __name__ == "__main__":
    run_experiments()
    print("\nAll datasets generated successfully!")