import json
import os

# Paths
RESULT_DATA_DIR = os.path.join("..", "result_data")
RESULTS_DIR = os.path.join("..", "results")
INPUT_FILENAME = "gold_only_model_results.jsonl"
INPUT_FILE = os.path.join(RESULT_DATA_DIR, INPUT_FILENAME)
REPORT_FILE = os.path.join(RESULTS_DIR, "evaluation_report.txt")

def evaluate_results(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    total_count = 0
    correct_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                gold = data["gold_answer"].strip()
                output = data["model_output"].strip()
                
                total_count += 1
                
                # Binary substring match check 
                if gold.lower() in output.lower():
                    correct_count += 1
                    
            except Exception as e:
                continue

    if total_count == 0:
        return None

    # Performance calculations
    accuracy = (correct_count / total_count) * 100
    incorrect_count = total_count - correct_count

    # Constructing plain text report
    report =  f"EXPERIMENT REPORT: {INPUT_FILENAME}\n"
    report += "=" * 40 + "\n"
    report += f"Total Samples:       {total_count}\n"
    report += "-" * 40 + "\n"
    report += f"Correct (Match):     {correct_count:<8} ({accuracy:.2f}%)\n"
    report += f"Incorrect/Failed:    {incorrect_count:<8} ({(incorrect_count/total_count)*100:.2f}%)\n"
    report += "=" * 40 + "\n"
    
    return report

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Run and Save
report_text = evaluate_results(INPUT_FILE)
if report_text:
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Success! Report saved to {REPORT_FILE}")
    print("\n" + report_text)