import json
import os
import re
import glob

# Paths
RESULT_DATA_DIR = os.path.join("..", "result_data")
RESULTS_DIR = os.path.join("..", "results")
FINAL_REPORT_FILE = os.path.join(RESULTS_DIR, "evaluation_file.txt")

def natural_sort_key(s):
    """ Sorts 'rand_2' before 'rand_10' by converting digits to integers. """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def parse_params(filename):
    """
    Returns (Gold, Distractor, Random, Retrieved)
    """
    # Gold format: gold_G_dist_D_rand_R
    gold_match = re.search(r"gold_(\d+)_dist_(\d+)_rand_(\d+)", filename)
    if gold_match:
        g, d, r = gold_match.groups()
        return g, d, r, "0"
    
    # BM25 format: kK_randR
    bm25_match = re.search(r"k(\d+)_rand(\d+)", filename)
    if bm25_match:
        k_val, rand_val = bm25_match.groups()
        return "0", "0", rand_val, k_val
        
    return "0", "0", "0", "0"

def evaluate_single_file(file_path):
    total_count = 0
    correct_count = 0
    if not os.path.exists(file_path): return None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                gold = data["gold_answer"].strip().lower()
                output = data["model_output"].strip().lower()
                total_count += 1
                if gold in output:
                    correct_count += 1
            except:
                continue

    if total_count == 0: return None
    accuracy = (correct_count / total_count) * 100
    return total_count, correct_count, accuracy

def format_table(title, files, columns):
    """
    columns: list of tuples (header_name, data_index)
    data_index map: 0:Gold, 1:Dist, 2:Rand, 3:Retrieved
    """
    if not files: return ""
    files.sort(key=natural_sort_key)
    
    # Build Header
    col_headers = " | ".join([f"{c[0]:<13}" for c in columns])
    header = f"{'Filename':<40} | {col_headers} | {'Total':<6} | {'Correct':<8} | {'Accuracy'}\n"
    
    table = f"\n{title}\n"
    table += "=" * len(header) + "\n"
    table += header
    table += "-" * len(header) + "\n"
    
    for file_path in files:
        fname = os.path.basename(file_path)
        params = parse_params(fname) # (G, D, R, K)
        stats = evaluate_single_file(file_path)
        
        if stats:
            total, correct, acc = stats
            param_str = " | ".join([f"{params[c[1]]:<13}" for c in columns])
            table += f"{fname:<40} | {param_str} | {total:<6} | {correct:<8} | {acc:.2f}%\n"
    
    table += "=" * len(header) + "\n"
    return table

def run_all_evaluations():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_files = glob.glob(os.path.join(RESULT_DATA_DIR, "*.jsonl"))
    
    if not result_files:
        print(f"No result files found in {RESULT_DATA_DIR}")
        return

    bm25_files = []
    gold_rand_files = []
    gold_dist_files = []

    for f in result_files:
        fname = os.path.basename(f)
        if "bm25" in fname:
            bm25_files.append(f)
        elif "dist_0" in fname:
            gold_rand_files.append(f)
        else:
            gold_dist_files.append(f)

    full_report = "SUMMARY EVALUATION REPORT\n"
    
    # Define which columns to show for each table
    full_report += format_table("BM25 EVALUATIONS", bm25_files, 
                                [("Retrieved(k)", 3), ("Random(r)", 2)])
    
    full_report += format_table("GOLD + RANDOM CONTEXT (Distractor=0)", gold_rand_files, 
                                [("Gold(g)", 0), ("Random(r)", 2)])
    
    full_report += format_table("GOLD + DISTRACTOR CONTEXT (Random=0)", gold_dist_files, 
                                [("Gold(g)", 0), ("Distractor(d)", 1)])

    with open(FINAL_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(full_report)
    
    print(full_report)
    print(f"Success! Report saved to {FINAL_REPORT_FILE}")

if __name__ == "__main__":
    run_all_evaluations()