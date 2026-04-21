# English-to-SQL with Llama 2

*A notebook-first Google Colab project for fine-tuning Llama-2-7B with LoRA and 4-bit quantization to generate SQL from natural-language questions and schema text.*

This repository is an end-to-end LLM fine-tuning project focused on a practical and interpretable use case: translating plain-English questions into SQL when the model is given the relevant table schema or DDL. It is designed as a strong educational and portfolio project rather than a production system. The core workflow lives in a single Colab-oriented notebook that covers dataset preparation, quantized model loading, PEFT/LoRA fine-tuning, lightweight inference, model saving and reloading, and a small Streamlit demo.

The most interesting part of the project is not just that it generates SQL, but how it does so efficiently. Instead of fully fine-tuning a 7B-parameter model, the notebook combines 4-bit quantized base weights with small trainable LoRA adapters so the experiment remains practical on a single Colab GPU. The demo then carries that work through to a usable UI that mirrors the training prompt format.

## Key Highlights

- Fine-tunes `NousResearch/Llama-2-7b-hf` for English-to-SQL generation.
- Uses **LoRA (PEFT)** for parameter-efficient adaptation instead of updating the full base model.
- Loads the base model in **4-bit NF4 quantization** with bitsandbytes to make a 7B model feasible in Colab.
- Trains on `ChrisHayduk/Llama-2-SQL-Dataset` using an instruction-style causal language modeling setup.
- Preserves **prompt-template alignment** between training and inference by using the same instruction/schema format in the Streamlit demo.
- Demonstrates the full lifecycle of a small LLM experiment: data loading, fine-tuning, inference, saving, reloading, and UI wrapping.
- Frames the work honestly as a notebook-first ML experiment, not a production-grade text-to-SQL platform.

## Demo / Screenshots

The notebook creates a Streamlit app inside Colab and exposes it with `cloudflared`, so the demo runs as a Colab-hosted UI rather than a separately packaged application.

![Streamlit demo screenshot](<Screenshot 2026-01-27 173311.png>)
![Streamlit demo screenshot](<Screenshot 2026-01-27 173140.png>)

## Table of Contents

- [Why This Project](#why-this-project)
- [What the Project Does](#what-the-project-does)
- [Core Ideas Demonstrated](#core-ideas-demonstrated)
- [How It Works](#how-it-works)
- [Project Workflow / Pipeline](#project-workflow--pipeline)
- [Model and Training Setup](#model-and-training-setup)
- [Prompt Format / Data Format](#prompt-format--data-format)
- [Inference Flow](#inference-flow)
- [Streamlit Demo](#streamlit-demo)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Setup / Installation](#setup--installation)
- [How to Run in Google Colab](#how-to-run-in-google-colab)
- [Training Configuration](#training-configuration)
- [Saving and Reloading the Fine-Tuned Model](#saving-and-reloading-the-fine-tuned-model)
- [Example Usage](#example-usage)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [What This Project Demonstrates](#what-this-project-demonstrates)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Why This Project

English-to-SQL is a compelling LLM use case because it sits at the intersection of natural language understanding and structured reasoning. A user expresses intent in plain English, but the output must conform to the syntax and structure of a database query. That makes it a useful problem for exploring where language models are strong, where they fail, and how prompt design plus lightweight fine-tuning can improve task-specific behavior.

This project is especially interesting because it is **schema-aware**. The model does not guess blindly from the question alone. Instead, it receives a description of the available tables and columns as part of the prompt, and it must generate SQL that is consistent with that provided context. That makes the task more grounded than generic text generation, while still being much simpler than building a live database agent.

From a portfolio perspective, the project demonstrates practical ML engineering tradeoffs:

- adapting a large language model without full fine-tuning,
- working within Colab hardware limits,
- aligning training and inference formats,
- and taking an experiment all the way to a simple interactive demo.

## What the Project Does

At a high level, this project fine-tunes Llama-2-7B so that it can:

1. read a plain-English request,
2. read a schema or DDL description of the relevant tables,
3. and generate a SQL query as the response.

This is **not** live database reasoning. The model does **not** connect to or inspect a real database. It only sees the text that the user provides in the prompt, especially the schema definition. In other words, the project is about **schema-aware prompting and fine-tuning**, not query execution or database validation.

The repository is intentionally notebook-centric:

- the main training and inference logic lives in `llama_fine_tuning_SQL_query_Bot.ipynb`,
- the Streamlit app is created from inside the notebook,
- and the intended runtime is Google Colab.

## Core Ideas Demonstrated

- **Parameter-efficient fine-tuning (LoRA):** the project adapts model behavior by training low-rank adapter weights instead of updating every base-model parameter.
- **Memory-efficient model loading:** the base Llama-2 model is loaded in 4-bit NF4 quantized form to reduce GPU memory pressure.
- **Causal LM fine-tuning on instruction-style data:** training examples are converted into a single continuation task where the model learns to produce SQL after a structured prompt.
- **Prompt-template alignment:** the Streamlit app builds prompts in the same format used during training, which helps keep inference behavior closer to the learned distribution.
- **Colab-first experimentation:** the workflow is designed around the practical constraints of a single Colab session, including training, saving to Google Drive, reloading, and demo serving.
- **Lightweight ML demo delivery:** the notebook creates a small UI so the fine-tuned model can be tested interactively without building a larger application stack.

## How It Works

The project follows a straightforward but technically meaningful pipeline:

1. Load a dataset of instruction-style English-to-SQL examples.
2. Load a Llama-2-7B base model in 4-bit quantized form.
3. Attach LoRA adapters to selected transformer projection layers.
4. Fine-tune the model on a 5,000-example subset of the training split.
5. Generate SQL by prompting the model with a question and schema text.
6. Save the fine-tuned artifacts to Google Drive.
7. Reload the saved model and serve it through a Colab-hosted Streamlit interface.

Under the hood, the notebook treats SQL generation as a causal language modeling task. Each datapoint already contains an `input` field and an `output` field, and the preprocessing step concatenates:

```text
input + output + <eos>
```

That combined sequence is tokenized and used for training. Conceptually, the model learns to continue an instruction prompt until it reaches the desired SQL answer.

### Training Strategy in Plain English

Rather than retraining a large model from scratch or fully updating all of its weights, this project uses a more practical adaptation strategy:

- keep the original Llama-2 weights mostly frozen,
- load those weights in a compressed 4-bit representation to save memory,
- train a small number of LoRA adapter parameters that steer the model toward SQL generation.

This combination is what makes a 7B model manageable in a Colab environment while still preserving the educational value of a real fine-tuning workflow.

### Why Prompt-Template Alignment Matters

One of the strongest design choices in the project is that inference matches training format closely. The Streamlit app does not send arbitrary free-form prompts. Instead, it builds the same structured template the model was fine-tuned on:

- `Below is an instruction...`
- `### Instruction:`
- `### Input:`
- `### Response:`

That alignment matters because fine-tuned models often behave best when the inference prompt looks like the data distribution they were trained on.

## Project Workflow / Pipeline

```text
Dataset load
  -> shuffle train split
  -> select 5,000 examples
  -> concatenate input + output + eos
  -> tokenize for causal LM
  -> load quantized Llama-2-7B
  -> attach LoRA adapters
  -> fine-tune with Hugging Face Trainer
  -> run lightweight inference check
  -> save model + tokenizer to Google Drive
  -> reload model from Drive
  -> launch Streamlit in Colab
  -> expose app with cloudflared
```

## Model and Training Setup

### Base Model

- `NousResearch/Llama-2-7b-hf`

### Dataset

- `ChrisHayduk/Llama-2-SQL-Dataset`
- The notebook loads the Hugging Face dataset, shuffles the `train` split, and selects **5,000 examples** for a smaller, faster fine-tuning run.

### Quantization Configuration

The base model is loaded with bitsandbytes 4-bit quantization:

- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_compute_dtype="float16"`

This is a practical memory-saving choice. The model remains usable for fine-tuning in Colab because the frozen base weights are stored more compactly.

### LoRA Configuration

The notebook configures LoRA through PEFT with:

- `r=16`
- `lora_alpha=32`
- `lora_dropout=0.05`
- `task_type="CAUSAL_LM"`

LoRA is applied to these transformer projection modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `up_proj`
- `down_proj`
- `gate_proj`

### Trainer Setup

Training is done with Hugging Face `Trainer` using:

- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4`
- `num_train_epochs=5`
- `learning_rate=3e-5`
- `fp16=True`
- `optim="paged_adamw_8bit"`
- `lr_scheduler_type="cosine"`
- `warmup_ratio=0.05`
- `output_dir="fine_tuning"`

The data collator is:

```python
DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

That is an important detail: this is standard **causal language model fine-tuning**, not masked-language modeling.

## Prompt Format / Data Format

The training data uses an instruction-style format where the question and schema context are packaged into a single prompt. A representative example looks like this:

```text
### Question:
Below is an instruction that describes a SQL generation task, paired with an input that provides further context about the available table schemas. Write SQL code that appropriately answers the request.

### Instruction:
What is the result on November 1, 1992?

### Input:
CREATE TABLE table_name_53 (result VARCHAR, date VARCHAR)

### Response:

### SQL:
```

In the notebook preprocessing step, the `input` text and the target `output` text are concatenated with the tokenizer EOS token before tokenization. That means the model is trained to continue the structured prompt with the expected SQL response.

## Inference Flow

After training, the notebook defines a helper:

```python
generate_sql(question: str)
```

Its behavior is simple and easy to follow:

1. tokenize the input prompt,
2. call `model.generate(...)`,
3. measure the original prompt length,
4. slice off the prompt tokens from the returned sequence,
5. stop at the first EOS token,
6. decode only the generated completion.

This keeps the output focused on the SQL continuation rather than returning the entire prompt again.

The notebook also performs a lightweight spot check using a shuffled sample from the dataset's `eval` split and compares the generated string with a reference SQL string. That comparison is useful as a quick demo sanity check, but it should not be interpreted as a rigorous measure of SQL correctness.

## Streamlit Demo

The repository does **not** contain a permanent checked-in `app.py`. Instead, the notebook writes the app dynamically with a `%%writefile app.py` cell inside Colab.

The demo is designed around the same structure used during fine-tuning:

- one text box for the English instruction,
- one text box for the schema / DDL,
- hidden prompt assembly that mirrors the training template,
- SQL generation displayed as the output.

### Demo Characteristics

- Runs inside Google Colab.
- Launches Streamlit on port `8501`.
- Exposes the local Streamlit server using `cloudflared`.
- Loads saved model artifacts from:

```text
/content/drive/MyDrive/llama_sql_finetune
```

- Caches the loaded model with `@st.cache_resource`.
- Lets the user configure:
  - model folder,
  - dtype,
  - `max_new_tokens`,
  - `do_sample`,
  - `temperature`,
  - `top_k`,
  - `top_p`.

This is best thought of as a **Colab-hosted demo UI** rather than a separately packaged deployment.

## Repository Structure

This is a notebook-first repository. The notebook is the main artifact, and the rest of the files mainly support documentation and presentation.

```text
.
├── llama_fine_tuning_SQL_query_Bot.ipynb
├── README.md
├── requirements.txt
├── Screenshot 2026-01-27 173140.png
├── Screenshot 2026-01-27 173311.png
└── LICENSE
```

### Notes on Structure

- `llama_fine_tuning_SQL_query_Bot.ipynb` contains the full workflow: installs, dataset loading, model setup, fine-tuning, inference, saving, reloading, and Streamlit app creation.
- `requirements.txt` lists the core training libraries used for the experiment.
- `streamlit` is installed later inside the notebook when the demo UI is created.
- There is no packaged Python module, no checked-in standalone app source, and no production deployment scaffold.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Accelerate
- PEFT / LoRA
- bitsandbytes
- TRL
- Streamlit
- Google Colab
- Google Drive
- cloudflared

## Setup / Installation

This project is designed primarily for **Google Colab**, not as a polished local application. The most reliable way to use it is to open the notebook in Colab and run the cells from top to bottom.

### Minimal Local Setup

```bash
git clone <your-repo-url>
cd llama-2-SQL-Query-Generator
pip install -r requirements.txt
```

Notes:

- `requirements.txt` covers the core model-training dependencies.
- The Streamlit demo dependencies are installed inside the notebook as part of the UI section.
- Because the project is Colab-first, local execution is secondary to the notebook workflow.

## How to Run in Google Colab

Open the notebook in Colab:

- [Open in Google Colab](https://colab.research.google.com/drive/1EtLZupv91ICpXHb-ptAxj8b1PVoYuxxC?usp=sharing)

Or upload `llama_fine_tuning_SQL_query_Bot.ipynb` to your own Colab environment.

### Recommended Flow

1. Open the notebook in Google Colab.
2. Run the dependency installation cells.
3. Load the dataset and prepare the 5,000-example training subset.
4. Load the quantized Llama-2-7B model and tokenizer.
5. Configure LoRA and fine-tune with Hugging Face `Trainer`.
6. Run the lightweight inference check.
7. Mount Google Drive and save the fine-tuned model artifacts.
8. Reload the saved model from Drive.
9. Run the Streamlit section to create `app.py`, start the app, and expose it with `cloudflared`.

## Training Configuration

The notebook uses the following training setup.

### Dataset and Preprocessing

- Dataset: `ChrisHayduk/Llama-2-SQL-Dataset`
- Training split: `dataset["train"]`
- Preprocessing:

```python
combined = x["input"] + x["output"] + tokenizer.eos_token
```

- Sampling strategy: shuffle the full training set, then select `range(5000)`

### Quantized Base Model

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)
```

### LoRA Setup

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "down_proj",
        "v_proj", "gate_proj", "o_proj", "up_proj"
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

### Trainer Arguments

```python
TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=3e-5,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    output_dir="fine_tuning"
)
```

## Saving and Reloading the Fine-Tuned Model

The intended workflow is:

1. fine-tune in Colab,
2. save to Google Drive,
3. reload from Drive for inference and demo usage.

The notebook mounts Google Drive and saves model artifacts to:

```text
/content/drive/MyDrive/llama_sql_finetune
```

It then includes a reload section that restores the tokenizer and model from that directory so the demo can use the saved version instead of the in-memory training session.

The notebook also includes commented-out code for optional Hugging Face Hub upload and Streamlit Space creation. Those sections are useful as extension ideas, but they are not the main workflow of the project.

## Example Usage

The Streamlit UI asks for two inputs:

- an English instruction,
- and a schema / DDL description.

Example:

```text
Instruction:
What is the result on November 1, 1992?

Input:
CREATE TABLE table_name_53 (result VARCHAR, date VARCHAR)
```

The app then internally builds a prompt that looks like:

```text
Below is an instruction that describes a SQL generation task, paired with an input that provides further context about the available table schemas. Write SQL code that appropriately answers the request.

### Instruction:
What is the result on November 1, 1992?

### Input:
CREATE TABLE table_name_53 (result VARCHAR, date VARCHAR)

### Response:
```

The model is expected to generate the SQL continuation for that prompt.

## Limitations

This project is intentionally honest about scope. It demonstrates a real fine-tuning workflow, but it is not a full text-to-SQL product.

- **No live database validation:** the model does not connect to a database or verify its SQL against actual data.
- **No execution-based evaluation:** the notebook does not run generated queries against a test database to measure correctness.
- **No formal benchmark suite:** evaluation is lightweight and based on a simple comparison against a reference string from the dataset's `eval` split.
- **Exact-match evaluation is limited:** SQL can have multiple correct forms, so string equality is not a strong measure of semantic correctness.
- **Schema-aware prompting only:** the model only sees the schema text provided in the prompt; it does not inspect live metadata or database state.
- **Notebook-first structure:** the workflow is built around a Colab notebook rather than a modular application or library.
- **No packaged deployment architecture:** there is no persistent checked-in app backend, API layer, containerization, or production serving setup.
- **Potential SQL hallucinations:** like other LLM systems, the model may invent columns, tables, or assumptions when the schema context is incomplete or ambiguous.
- **Dialect ambiguity:** SQL syntax can vary across SQLite, PostgreSQL, MySQL, and other engines.
- **Long schema inputs can be difficult:** larger schema descriptions may strain the prompt budget and reduce output quality.

## Future Improvements

There are several realistic directions to take this project further:

- Add **execution-based evaluation** using a real test database and answer validation.
- Introduce stronger **experiment tracking** and reproducibility practices.
- Convert the notebook workflow into a more structured Python package or application.
- Add safer and more explicit **SQL-generation constraints** before execution.
- Support more complex scenarios such as **multiple related schemas** or richer database context.
- Improve prompt and output validation for malformed or incomplete SQL.
- Expand evaluation beyond exact match to more meaningful SQL correctness checks.
- Separate training, inference, and demo logic into cleaner reusable components.
- Make the optional Hugging Face Hub / Space path more reproducible and better documented.

## What This Project Demonstrates

For AI / ML / LLM engineering roles, this project demonstrates practical experience with:

- fine-tuning a large language model for a task-specific use case,
- using **PEFT / LoRA** instead of full-parameter training,
- using **4-bit quantization** to work within limited GPU memory,
- building a Hugging Face training workflow in Colab,
- preparing instruction-style data for causal LM fine-tuning,
- aligning inference prompts with the model's training format,
- saving and reloading fine-tuned model artifacts,
- wrapping an ML experiment in a small interactive Streamlit UI,
- making technically grounded tradeoffs without overstating scope or results.

It is best viewed as a thoughtful, end-to-end learning project that shows hands-on familiarity with modern LLM adaptation techniques and pragmatic experimentation workflows.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Hugging Face Transformers
- Hugging Face Datasets
- PEFT / LoRA
- bitsandbytes
- Streamlit
- Dataset: `ChrisHayduk/Llama-2-SQL-Dataset`

