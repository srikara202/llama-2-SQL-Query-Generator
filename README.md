# Llama-2-7B → English-to-SQL (Colab fine-tune + local Streamlit demo)

This repo/notebook shows how I fine-tuned **Llama-2-7B** to generate **SQL queries** from:

1. a question in plain English, and
2. a description of the available table(s) (usually as **SQL DDL**).

Everything here was built to run comfortably in **Google Colab**:

* training is done in Colab,
* the Streamlit UI is launched **inside Colab** (not hosted on Hugging Face Spaces),
* the model is trained using **LoRA adapters** + **bitsandbytes 4‑bit loading** so it fits on a single GPU.

---

## What you get

* A Colab notebook that:

  * loads a SQL dataset,
  * loads Llama-2-7B in **4-bit** (bitsandbytes),
  * adds **LoRA** adapters (PEFT),
  * fine-tunes on a small sample,
  * and runs a **Streamlit** demo to type a question + schema and get SQL back.

---

## How it works (high-level)

### 1) We start with a big model… and avoid updating all of it

Llama-2-7B has billions of parameters. Fully fine-tuning all weights is expensive (memory + time).

Instead, we do **parameter-efficient fine-tuning**:

* We **freeze** the original model weights.
* We train a small set of extra weights (LoRA) that “nudge” the model toward SQL generation.

### 2) We also load the frozen weights in 4-bit (bitsandbytes)

Even if we freeze the base weights, they still need to sit in GPU memory for forward/backprop.
So we load them in **4-bit** to reduce memory pressure.

This combo is why this project works nicely in Colab.

---

## LoRA in plain English (the “low-rank chunks” idea)

A transformer is basically a stack of huge matrix multiplications.
For a single linear layer, you can think of it as:

* You have a weight matrix **W** (size `d_out × d_in`).
* The layer computes `y = x · Wᵀ` (or equivalently `W x`, depending on convention).

If you fully fine-tune, you update **all entries** of **W**.

**LoRA** says: don’t touch **W**. Instead, learn a small update **ΔW** and add it on top:

* `W' = W + ΔW`

But we don’t learn ΔW as a full matrix (that would still be huge). We force it to be **low-rank**.
That means we represent it as the product of two much smaller matrices:

* `ΔW = B · A`

where:

* `A` has shape `r × d_in`
* `B` has shape `d_out × r`
* `r` is a small number like 8, 16, 32… (**much smaller** than `d_in` and `d_out`)

So instead of learning `d_out × d_in` parameters, you learn:

* `d_out × r` + `r × d_in`

That’s the “split into low-ranked multipliable chunks” idea.
You only train **A** and **B**, and you keep the original **W** frozen.

Why it works in practice:

* Many tasks don’t need a complete rewrite of the model weights.
* A low-rank update is often enough to steer behavior (here: produce SQL more reliably).

---

## bitsandbytes in plain English (what 4-bit loading means)

bitsandbytes is a library that helps run large models on smaller GPUs by:

1. **Quantizing weights** (storing them with fewer bits), and
2. offering memory-friendly optimizers (like 8-bit Adam variants).

### The key idea

Normally, model weights are stored in **16-bit** or **32-bit** floats.

With 4-bit quantization:

* weights are stored in **4 bits** (tiny),
* but when the GPU actually does a matrix multiply, the weights are **de-quantized on the fly** into a compute type (in this project, float16).

So you get:

* **big memory savings** for storage,
* while still doing compute in a format the GPU likes.

### NF4 (what it is, conceptually)

In this notebook I use **NF4** (NormalFloat4).
You can think of it as a smarter 4-bit “bucket system” for values:

* instead of evenly spaced 4-bit levels,
* NF4 uses levels that better match how real neural network weights are distributed.

The practical outcome is: better accuracy compared to naive 4-bit rounding.

### Important note

In this project the **base model weights** are loaded in 4-bit, and **LoRA adapters** are trained on top.
This is why training stays feasible in Colab.

---

## Dataset

* Dataset used: `ChrisHayduk/Llama-2-SQL-Dataset`
* In the notebook, I shuffle the training split and pick a **subset of 5,000 examples** for a quick fine-tune.

The dataset already contains instruction-style strings:

* `input` (prompt that includes the question + schema)
* `output` (the target SQL)

During preprocessing, I concatenate:

```
combined_text = input + output + <eos>
```

and tokenize it for causal language modeling.

---

## Prompt format (what the model expects)

The dataset uses an instruction style like this (example):

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

When you use the Streamlit app, you’ll fill in the **Instruction** and **Input** fields and the app will assemble the prompt.

---

## What’s inside (repo / notebook)

This project is centered around a Colab notebook:

* `llama_fine_tuning_SQL_query_Bot.ipynb`

  * installs deps
  * loads dataset
  * loads Llama-2-7B in 4-bit
  * applies LoRA
  * trains with Transformers Trainer
  * runs inference + Streamlit demo

---

## Run it in Google Colab

### 1) Open the notebook

Upload/open:

* `llama_fine_tuning_SQL_query_Bot.ipynb`

Then run cells top-to-bottom.

---

### 2) Training configuration (the actual settings used)

**Base model**

* `NousResearch/Llama-2-7b-hf`

**bitsandbytes 4-bit config**

* `load_in_4bit=True`
* `bnb_4bit_quant_type="nf4"`
* `bnb_4bit_compute_dtype="float16"`

**LoRA (PEFT) config**

* `r=16`
* `lora_alpha=32`
* `lora_dropout=0.05`
* `target_modules=['q_proj','k_proj','down_proj','v_proj','gate_proj','o_proj','up_proj']`
* `task_type="CAUSAL_LM"`

**Trainer hyperparameters**

* `per_device_train_batch_size=1`
* `gradient_accumulation_steps=4`
* `num_train_epochs=5`
* `learning_rate=3e-5`
* `fp16=True`
* `optim="paged_adamw_8bit"`
* `lr_scheduler_type="cosine"`
* `warmup_ratio=0.05`
* `output_dir="fine_tuning"`

---

## Running the Streamlit app in Colab

The notebook includes a Streamlit UI so you can type:

* an English instruction
* a schema / DDL

…and get SQL back.

### Typical flow

1. Install Streamlit + any tunnel tool you like (ngrok / cloudflared / localtunnel).
2. Launch Streamlit on a port.
3. Expose it to the browser using a public URL.

Because Colab doesn’t behave like a normal local machine, you usually need a tunnel.
A common approach is:

* run `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
* use `cloudflared` to expose `localhost:8501`

Click the "https://....trycloudflare.com" link in the output to access the streamlit application

> The exact tunnel commands depend on what you prefer. For the exact setup I used, follow the notebook steps.

---

## Inference settings (defaults used in the notebook)

The demo uses these generation defaults:

* `max_new_tokens = 80`
* `temperature = 0.7`
* `top_p = 0.9`
* `top_k = 50`
* `do_sample = False`

---

## Notes on quality (and what to expect)

This project is meant as a practical, end-to-end demo.
A few honest notes:

* The dataset is helpful, but SQL is tricky: there can be many correct answers for the same question.
* Exact string match is not a great metric.
* The best evaluation is **execution-based** (run the SQL on a test database and check correctness).

---

## Limitations

* **Hallucinated columns/tables** can happen if the schema is unclear or if the question implies missing fields.
* **SQL dialect ambiguity**: SQLite vs Postgres vs MySQL differences (dates, quotes, functions).
* **Long schemas** can overflow the context window, leading to truncated input.
* Always **review SQL** before running it on important data.

---

## Acknowledgements

* Hugging Face Transformers
* PEFT (LoRA)
* bitsandbytes
* Streamlit
* Dataset: `ChrisHayduk/Llama-2-SQL-Dataset`

---

## License

See `LICENSE`.
