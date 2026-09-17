# dl-project3-notebooks

**Small-model Chinese dialog LLM fine-tuning — dataset prep, full fine-tune, and a Gradio chat demo.**

Notebooks for the third project of a Deep Learning course at NKUST. The goal
is to full-fine-tune a small Chinese causal LM (BLOOM-389M, roughly 400 M
parameters) on a 15.7k-example aspect-based hotel review dataset, then serve
the fine-tuned checkpoint through a Gradio chat UI with automatic
Simplified <-> Traditional Chinese conversion.

The repository intentionally keeps the three stages — **dataset preparation**,
**training**, and **inference / demo** — as separate notebooks so each stage
can be re-run in isolation.

---

## Overview

The project pipeline:

```
  hotel-review CSV (16k Traditional-Chinese aspect-labelled reviews)
                 |
                 v
  (資料集整理)   convert each row to an Alpaca-style
                 {instruction, input, output} record,
                 opencc-translate to Simplified Chinese,
                 tokenise with BLOOM tokenizer,
                 append </s> EOS to every training example,
                 mask the user-prompt portion of `labels` with -100
                 |
                 v
                 datasets saved to ./train_dataset/data_train + data_val
                 |
                 v
  (模型訓練)     load Langboat/bloom-389m-zh (or YeungNLP/bloomz-396m-zh),
                 full fine-tune for 1 epoch on a T4 GPU
                 (BATCH_SIZE=64, MICRO_BATCH=4, grad-accum=16, LR=8e-6),
                 save to ./my-pretrained-3epochs-YeungNLP-zh-ch
                 |
                 v
  (test.ipynb)   Gradio ChatInterface: user types Traditional Chinese,
                 t2s converts to Simplified for the model,
                 model generates, s2t converts back to Traditional
```

The task the model learns is **aspect-based sentiment classification phrased
as instruction following**: given a hotel review, the assistant answers which
aspects are mentioned (cleanliness, facilities, service, location,
value-for-money, other, overall) and whether each is positive / neutral /
negative.

---

## Notebook list

| Folder / Notebook | Role |
|-------------------|------|
| `資料集整理/10-14-tokenize尾部加eos_token-Human短提示語.ipynb` | **Dataset preparation.** Reads `hotel-review-飯店留言合併1.6萬筆(資料集).csv`, converts each row into `{instruction, input, output}` JSON, uses `opencc` to translate Traditional Chinese to Simplified, then tokenises with `Langboat/bloom-389m-zh`'s `BloomTokenizerFast`. Explicitly appends `tokenizer.eos_token_id` to `input_ids` (matching the notebook title "add eos_token at the tail"), copies `input_ids` into `labels`, and masks the `Human:` prompt span with `-100` so loss is only computed on the assistant reply. Saves an Arrow dataset to `./train_dataset/data_train` and `./train_dataset/data_val` via a 98.5/1.5 train/val split. Also demonstrates left-padding batching with `DataCollatorForSeq2Seq`. |
| `模型訓練/w15_10_全微調_YeungNLP簡體.ipynb` | **Full fine-tune (全微調).** Colab-flavoured training notebook. Loads either `Langboat/bloom-389m-zh` or `YeungNLP/bloomz-396m-zh` from Hugging Face, mounts Google Drive for storage, and runs a Hugging Face `Trainer` on the prepared dataset for 1 epoch. Key hyper-params: `per_device_train_batch_size=4`, `gradient_accumulation_steps=16` (effective batch 64), `learning_rate=8e-6`, `warmup_ratio=0.05`, `weight_decay=1e-5`, `optim="adamw_torch"`, checkpoints every 20 steps with `save_total_limit=2`. Reports ~28 min wall time for one epoch on a Colab T4 GPU, final train loss ~0.13 on 15,786 training examples. Saves the checkpoint to `my-pretrained-3epochs-YeungNLP-zh-ch`. |
| `test.ipynb` | **Inference / Gradio chat demo.** Reloads the fine-tuned model with `AutoModelForCausalLM.from_pretrained('./my-pretrained-3epochs-YeungNLP-zh-ch')`, wraps it in a `TextIteratorStreamer`-style predict loop (`max_length=400`, `top_p=0.95`, `top_k=200`, `temperature=1.0`, `repetition_penalty=1.2`, `StopOnTokens` custom `StoppingCriteria`) and builds a `gradio.ChatInterface`. Handles Traditional/Simplified conversion via `opencc` (`t2s` for input, `s2t` for output) so users can chat in Traditional Chinese even though the model was trained on Simplified. |

---

## Dataset organization rationale

The `資料集整理` (dataset organisation) folder is separated from `模型訓練`
(model training) on purpose:

1. **Reproducibility.** Tokenisation is deterministic but tied to a specific
   tokenizer (`Langboat/bloom-389m-zh`, which shares vocab with
   `YeungNLP/bloomz-396m-zh`). Running the preparation step once and saving
   an `Arrow` dataset means the training notebook does not have to re-tokenise
   ~16k examples on every restart.
2. **Loss masking correctness.** For instruction tuning we only want the
   model to learn the assistant response, not to reproduce the user prompt.
   The prep notebook computes the length of the user-prompt tokens and
   overwrites that prefix of `labels` with `-100`. Doing this at data-prep
   time (rather than during training) keeps the collator simple —
   `DataCollatorForSeq2Seq` just left-pads with `-100` for `labels` and `0`
   for `input_ids`.
3. **Trad/Simp handling.** BLOOM-zh models were pre-trained largely on
   Simplified Chinese, so the raw Traditional-Chinese hotel reviews are
   converted with `opencc('t2s')` before tokenisation. The inference notebook
   reverses this at generation time so the end-user still sees Traditional
   output.
4. **Alpaca-format staging.** The prep notebook emits an intermediate
   `output_results.json` / `output_results_simplified.json` in the standard
   `{instruction, input, output}` schema, which makes the same corpus
   trivially reusable by other instruction-tuning stacks (LoRA, QLoRA,
   Alpaca-LoRA, etc.).

### Dataset file

- `資料集整理/hotel-review-飯店留言合併1.6萬筆(資料集).csv` —
  ~16,000 Traditional-Chinese hotel reviews. Per-row columns include
  `comment_text` and seven aspect-sentiment columns (`整潔舒適情緒`,
  `設施情緒`, `服務情緒`, `地點情緒`, `性價比情緒`, `其他情緒`,
  `整體情緒`), each encoded as `0 = not mentioned`, `1 = positive`,
  `2 = neutral`, `3 = negative`.

---

## Tech stack

- **Python 3.10** on Colab / miniconda (`ai23` env in the notebook logs)
- **PyTorch 2.x** with CUDA (Colab T4 during training, CPU-fallback for the demo)
- **Hugging Face** — `transformers==4.32`, `datasets`, `accelerate`
- **Base model** — [`Langboat/bloom-389m-zh`](https://huggingface.co/Langboat/bloom-389m-zh)
  (primary) or [`YeungNLP/bloomz-396m-zh`](https://huggingface.co/YeungNLP/bloomz-396m-zh)
  (float16 variant)
- **Chinese conversion** — `opencc` (`t2s.json` / `s2t.json`)
- **UI** — `gradio` (ChatInterface with retry / undo / clear buttons)

There is no `requirements.txt`; each notebook installs what it needs via
`!pip install` in the first few cells.

---

## Running

```bash
# 1. Clone
git clone https://github.com/Elainedu/dl-project3-notebooks.git
cd dl-project3-notebooks

# 2. (Recommended) create env
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install deps
pip install "transformers==4.32" datasets accelerate torch opencc gradio pandas
```

Then, in order:

1. Open `資料集整理/10-14-tokenize尾部加eos_token-Human短提示語.ipynb`,
   update the CSV path near the top, and run all cells. This writes
   `./train_dataset/data_train/` and `./train_dataset/data_val/`.
2. Open `模型訓練/w15_10_全微調_YeungNLP簡體.ipynb` (Colab-oriented; skip
   the `drive.mount` cell if running locally), point `train_data_path` at
   the folder produced in step 1, and run. The trained checkpoint lands in
   `./my-pretrained-3epochs-YeungNLP-zh-ch/`.
3. Open `test.ipynb`, ensure `model_name_or_path` matches the checkpoint
   folder above, and run all cells. Gradio prints a local URL
   (typically `http://127.0.0.1:7862`).

### Hardware notes

- Training on a T4 (Colab free tier) fits at MICRO_BATCH_SIZE=4 without OOM;
  enable `model.gradient_checkpointing_enable()` if you see OOM.
- Inference runs on CPU (slowly) — the notebook auto-detects CUDA and falls
  back to CPU otherwise.

---

## Repository layout

```
dl-project3-notebooks/
├── README.md                                              # this file
├── .gitignore
├── test.ipynb                                             # Gradio chat demo
├── 模型訓練/
│   └── w15_10_全微調_YeungNLP簡體.ipynb                    # full fine-tune
└── 資料集整理/
    ├── 10-14-tokenize尾部加eos_token-Human短提示語.ipynb   # dataset prep
    └── hotel-review-飯店留言合併1.6萬筆(資料集).csv        # raw source data
```

---

## License

MIT. For academic / coursework use.
