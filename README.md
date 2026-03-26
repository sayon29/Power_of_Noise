# Power Of Noise

This repository contains the implementation and experimental setup based on the research paper **"The Power of Noise: Redefining Retrieval for RAG Systems"**.The study explores how the characteristics of retrieved documents—specifically their relevance, position, and the presence of "noise"—impact the effectiveness of Retrieval-Augmented Generation (RAG) systems.

---

### Experiment Overview

The core of this experiment is a systematic examination of how different types of passages provided to a Large Language Model (LLM) affect its accuracy in an Open-Domain Question Answering (OpenQA) task. 

Key findings from the study include:
* **The Negative Impact of Distractors:** Top-scoring documents retrieved by an Information Retrieval (IR) system that do not contain the actual answer (distracting documents) can significantly degrade LLM performance.
* **The "Power of Noise":** Counter-intuitively, adding completely random documents (informational noise) to the prompt can improve LLM accuracy by up to **35%**
* **Positioning Matters:** Relevant information is most effective when placed **near the query**; models struggle when the "gold" document is placed in the middle of a long context.

---

### Setup Instructions

Follow these steps to prepare the environment and generate the data required for the experiments.

#### 1. Data Preparation
This project utilizes the **2WikiMultiHopQA** dataset. 
* **Download:** You can download the dataset files from this [Dropbox link](https://www.dropbox.com/preview/(Copy)%20data_ids.zip?path=&scs=true).
* **Create the Dataset Folder:** Create a directory named `dataset` in the root of your project.
* **Organize:** Place the downloaded files (specifically `dev.json`) into the `dataset/` folder.

#### 2. Output Directory
* **Create the Prompts Folder:** Create a directory named `prompts` in the root of the project. 
* **Note:** This folder is included in the `.gitignore` to prevent large generated files from being pushed to GitHub.

#### 3. Result_Data Directory
* **Create the Result_data Folder:** Create a directory named `result_data` in the root of the project, store the results from kaggle here. 
* **Note:** This folder is included in the `.gitignore` to prevent large generated files from being pushed to GitHub.

#### 4. Results Directory
* **Create the Results Folder:** Create a directory named `results` in the root of the project, the evaluations are stored here. 

#### 5. Generate Prompts
Once the data is in place, run the generation script from the root directory to create the formatted JSONL file used for testing, this JSON file will be saved in prompts folder.

```bash
python src/gold_prompt.py
```

#### 6. Run on Kaggle
First get the models you want from hugging face and get the HF_TOKEN, save it in add-ons in a kaggle notebook (default version is llama2 for this jupyter notebook). Load the dataset in kaggle and name it gold_prompt. Run the script on kaggle.

#### 7. Evaluate Results
Run the evaluation.py file for getting accuracy, run it with proper filenames, the results will be saved in the results folder

```bash
python src/evaluation.py
```