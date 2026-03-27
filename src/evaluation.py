import json
import os
import re
import glob

# Paths
RESULT_DATA_DIR = os.path.join("..", "result_data")
RESULTS_DIR = os.path.join("..", "results")
FINAL_REPORT_FILE = os.path.join(RESULTS_DIR, "evaluation_file.txt")

def parse_params(filename):
    """
    Extracts parameters from filenames like 'bm25_k4_rand8.jsonl' 
    or 'gold_2_dist_0_rand_4.jsonl'
    """
    # Check for original format
    original = re.search(r"gold_(\d+)_dist_(\d+)_rand_(\d+)", filename)
    if original:
        return original.groups()
    
    # Check for BM25 format: bm25_k4_rand8
    retriever = re.search(r"k(\d+)_rand(\d+)", filename)
    if retriever:
        k_val, rand_val = retriever.groups()
        return "0", k_val, rand_val # Gold=0, Retrieved=K, Random=Rand
        
    return "N/A", "N/A", "N/A"

def evaluate_single_file(file_path):
    total_count = 0
    correct_count = 0

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                gold = data["gold_answer"].strip().lower()
                output = data["model_output"].strip().lower()
                
                total_count += 1
                # Binary substring match check 
                if gold in output:
                    correct_count += 1
            except:
                continue

    if total_count == 0:
        return None
        
    accuracy = (correct_count / total_count) * 100
    return total_count, correct_count, accuracy

def run_all_evaluations():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Find all result files
    result_files = sorted(glob.glob(os.path.join(RESULT_DATA_DIR, "*.jsonl")))
    
    if not result_files:
        print(f"No result files found in {RESULT_DATA_DIR}")
        return

    full_report = "SUMMARY EVALUATION REPORT (Binary Substring Match)\n"
    full_report += "=" * 90 + "\n"
    # Table Header
    full_report += f"{'Filename':<40} | {'G':<2} | {'K':<2} | {'R':<2} | {'Total':<6} | {'Correct':<8} | {'Accuracy'}\n"
    full_report += "-" * 90 + "\n"

    for file_path in result_files:
        fname = os.path.basename(file_path)
        g, k, r = parse_params(fname)
        
        stats = evaluate_single_file(file_path)
        if stats:
            total, correct, acc = stats
            full_report += f"{fname:<40} | {g:<2} | {k:<2} | {r:<2} | {total:<6} | {correct:<8} | {acc:.2f}%\n"

    full_report += "=" * 90 + "\n"

    # Save and Print
    with open(FINAL_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(full_report)
    
    print(full_report)
    print(f"Success! Report saved to {FINAL_REPORT_FILE}")

if __name__ == "__main__":
    run_all_evaluations()