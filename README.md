# Power Of Noise

This repository contains the implementation and experimental setup based on the research paper **"The Power of Noise: Redefining Retrieval for RAG Systems"**.The study explores how the characteristics of retrieved documents—specifically their relevance, position, and the presence of "noise"—impact the effectiveness of Retrieval-Augmented Generation (RAG) systems.

---

### Experiment Overview

The core of this experiment is a systematic examination of how different types of passages provided to a Large Language Model (LLM) affect its accuracy in an Open-Domain Question Answering (OpenQA) task[cite: 72, 109]. 

Key findings from the study include:
* **The Negative Impact of Distractors:** Top-scoring documents retrieved by an Information Retrieval (IR) system that do not contain the actual answer (distracting documents) can significantly degrade LLM performance.
* **The "Power of Noise":** Counter-intuitively, adding completely random documents (informational noise) to the prompt can improve LLM accuracy by up to **35%**
* **Positioning Matters:** Relevant information is most effective when placed **near the query**; models struggle when the "gold" document is placed in the middle of a long context, often referred to as being "lost in the middle".

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

#### 3. Generate Prompts
Once the data is in place, run the generation script from the src folder to create the formatted JSONL file used for testing, this JSON file will be saved in prompts folder.

```bash
python src/gold_prompt.py