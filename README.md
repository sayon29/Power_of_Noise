# Power Of Noise

This repository contains the implementation and experimental pipeline based on the research paper:

**"The Power of Noise: Redefining Retrieval for RAG Systems"**

The project studies how retrieved document properties—such as relevance, ordering, and noise—affect the performance of Retrieval-Augmented Generation (RAG) systems in Open-Domain Question Answering (OpenQA).

---

## Experiment Overview

The experiments analyze how different types of documents provided to a Large Language Model influence answer accuracy.

Key observations:
- Distractor documents negatively impact performance
- Adding random noise can significantly improve accuracy
- Positioning of relevant documents affects model performance

---

## Setup and Workflow

Follow the steps below to reproduce the full pipeline.

---

## 1. Data Preparation

This project utilizes the **2WikiMultiHopQA** dataset.

- **Download:** You can download the dataset files from this [Dropbox Link](https://www.dropbox.com/preview/(Copy)%20data_ids.zip?path=&scs=true)  
- **Create the Dataset Folder:** Create a directory named `dataset` in the root of your project  
- **Organize:** Place the downloaded files (specifically `dev.json`) into the `dataset/` folder  

---

## 2. Directory Setup

Create the following directories in the root of your project:

```
dataset/        # Contains dev.json and generated random_doc_pool.json
prompts/        # Generated prompt files
result_data/    # Raw outputs from Kaggle runs
results/        # Final evaluation outputs
```

---

## 3. Dataset Validation

Run the statistics script to verify dataset integrity and filtering:

```bash
python src/stats.py
```

---

## 4. Generate Random Document Pool

Generate a pool of random documents used for noise injection:

```bash
python src/random_doc_pool_gen.py
```

This creates:

```
dataset/random_doc_pool.json
```

---

## 5. Generate Prompts

Two pipelines are used to generate prompts.

### 5.1 Gold-Based Prompt Generation

This pipeline uses:
- Exactly **2 gold documents**
- Optional distractors
- Random documents from the pool  

Parameters such as the number of distractors and random documents can be modified in `create_all_prompts_gold.py`.

Run:

```bash
python src/create_all_prompts_gold.py
```

This internally calls:
- `prompt_gen_gold.py`

---

### 5.2 Retriever-Based Prompt Generation (BM25)

This pipeline:
- Uses a **BM25 retriever** to fetch top-k relevant documents
- Adds random noise documents
- Does **not** use gold documents  

Parameters such as `k` (top retrieved documents) and the number of random documents can be modified in `create_all_prompts_retriever.py`.

Run:

```bash
python src/create_all_prompts_retriever.py
```

This internally uses:
- `prompt_gen_retriever.py`
- BM25 retrieval over the document pool

---

### Important Constraints

- Gold documents are strictly **limited to exactly 2**
- Distractor documents are limited to a **maximum of 8** for the WikiMultiHop dataset
- Answer length is constrained to a maximum of **5 tokens**
- Total samples are capped using the `LIMIT` parameter in the scripts

Generated prompt files are stored in:

```
prompts/
```

---

## 6. Run Experiments on Kaggle

### Steps

1. Create a new Kaggle notebook  
2. Upload the `prompts/` folder as a dataset  
   - Dataset name should be: `prompts`  
3. Add your Hugging Face token (`HF_TOKEN`) in Kaggle secrets  
4. Load the dataset inside the notebook  
5. Run the provided notebook (`power-of-noise-wikimultihop.ipynb`)  
6. Note that change % of the first cell command of %pip install to !pip install when running in kaggle

### Model Configuration

- Default model used: LLaMA2  
- Models are loaded using **4-bit quantization**  
- Uses Hugging Face Transformers  

### Important

- Select the correct **prompt index range** using:
  - `START_FILE_INDEX`
  - `END_FILE_INDEX`  
- Ensure the correct prompt file is being evaluated  

---

## 7. Collect Results

After execution:

- Download the outputs from Kaggle  
- Place them in:

```
result_data/
```

---

## 8. Evaluate Results

- Evaluation uses : Substring match (Finds if actual answer is a substring of model output) 

Run:

```bash
python src/evaluation.py
```

Outputs will be saved in:

```
results/
```

---

## Summary

This pipeline enables experimentation over:

- Effect of distractors  
- Impact of random noise  
- Document ordering and positioning  

The goal is to better understand how retrieval behavior influences RAG system performance.