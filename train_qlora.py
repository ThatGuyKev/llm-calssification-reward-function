"""QLoRA fine-tune Qwen3-Embedding as a 3-way classifier (A wins / B wins / tie).

Defaults target one H100 80 GB: Qwen3-Embedding-0.6B, max-len 2048, length-grouped batches, head+tail truncation,
zero-init classification head with 10x LR. Add --no-4bit for bf16 LoRA instead of QLoRA.
Falls back to fp16 + GradScaler on GPUs without bf16.

Examples:
  python train_qlora.py --smoke                                  # tiny end-to-end check
  python train_qlora.py --no-4bit                                # real run, bf16 LoRA
  python train_qlora.py --model Qwen/Qwen3-Embedding-4B --no-4bit --max-len 1024 --bs 8 --accum 4 --out outputs/qwen3-emb-4b
  python train_qlora.py --predict-only                           # reuse saved adapter for test preds
"""
import argparse, json, math, os, random, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="any HF decoder LLM, e.g. Qwen/Qwen3-Embedding-4B, Qwen/Qwen3-Embedding-8B, tiiuae/Falcon3-7B-Base")
p.add_argument("--no-4bit", action="store_true", help="bf16 LoRA instead of QLoRA (fits a 9B on an H100 80 GB, a bit better quality)")
p.add_argument("--head-lr-mult", type=float, default=10.0, help="LR multiplier for the classification head")
p.add_argument("--data", default="data/llm-classification-finetuning")
p.add_argument("--out", default="outputs/qwen3-emb-0.6b")
p.add_argument("--max-len", type=int, default=2048)
p.add_argument("--n-train", type=int, default=0, help="0 = all")
p.add_argument("--val-frac", type=float, default=0.05)
p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--accum", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--r", type=int, default=16)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--grad-ckpt", action="store_true", help="trade ~30%% speed for memory (off by default on 80 GB)")
p.add_argument("--attn", default="sdpa", help="sdpa | eager | flash_attention_2. Gemma-2 needs eager or flash_attention_2 (sdpa skips logit soft-capping)")
p.add_argument("--pred-bs", type=int, default=32)
p.add_argument("--smoke", action="store_true")
p.add_argument("--predict-only", action="store_true")
a = p.parse_args()
if a.smoke:
    a.n_train, a.epochs, a.accum, a.max_len = 256, 1.0, 2, 256

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.model)
PAD = tok.pad_token_id
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
DT = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16  # Turing "supports" bf16 only via slow emulation
print("compute dtype:", DT, flush=True)

# ---------- data ----------
def parse(s):
    try:
        return " ".join(x if x is not None else "" for x in json.loads(s))
    except Exception:
        return str(s)

H_P, H_A, H_B = [tok.encode(s, add_special_tokens=False) for s in ("<prompt>: ", "\n\n<response_a>: ", "\n\n<response_b>: ")]
H_END = tok.encode("\n\n<winner>:", add_special_tokens=False)

PRE = [tok.bos_token_id] if tok.bos_token_id is not None else []  # Gemma/Llama expect BOS

def head_tail(x, n):
    """Keep the first 2/3 and last 1/3 of the budget: responses often end with the conclusion."""
    if len(x) <= n: return x
    h = (2 * n) // 3
    return x[:h] + x[len(x) - (n - h):] if n - h > 0 else x[:h]

def encode(prompt, ra, rb, max_len):
    """Fixed budget: prompt<=25%, remaining split evenly; short fields give slack to long ones."""
    overhead = len(PRE) + len(H_P) + len(H_A) + len(H_B) + len(H_END) + 1
    budget = max_len - overhead
    P, A, B = prompt, ra, rb  # already token ids (see build)
    pb = min(len(P), budget // 4)
    rest = budget - pb
    ab = min(len(A), max(rest // 2, rest - len(B)))
    bb = min(len(B), rest - ab)
    ab = min(len(A), rest - bb)
    return PRE + H_P + head_tail(P, pb) + H_A + head_tail(A, ab) + H_B + head_tail(B, bb) + H_END + [tok.eos_token_id]

def tok_all(texts, max_len):
    # char pre-truncation keeps the fast tokenizer cheap; ~4 chars/token so max_len*5 chars is plenty
    texts = [t[: max_len * 5] for t in texts]
    return tok(texts, add_special_tokens=False)["input_ids"]

def build(df, max_len, with_label=True):
    pr, ra, rb = (tok_all(df[c].map(parse).tolist(), max_len) for c in ("prompt", "response_a", "response_b"))
    ys = np.argmax(df[["winner_model_a", "winner_model_b", "winner_tie"]].values, 1) if with_label else None
    items = []
    for i in range(len(df)):
        it = {"x": encode(pr[i], ra[i], rb[i], max_len), "xs": encode(pr[i], rb[i], ra[i], max_len)}  # xs = swapped
        if with_label:
            it["y"] = int(ys[i])
        items.append(it)
    return items

def collate(batch, key="x", swap_labels=False):
    L = max(len(b[key]) for b in batch)
    ids = torch.full((len(batch), L), PAD); mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b[key])] = torch.tensor(b[key]); mask[i, :len(b[key])] = 1
    out = {"input_ids": ids, "attention_mask": mask}
    if "y" in batch[0]:
        out["labels"] = torch.tensor([b["y"] for b in batch])
    return out

SWAP = torch.tensor([1, 0, 2])

# ---------- model ----------
def load_base():
    bnb = None if a.no_4bit else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DT)
    m = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=3, quantization_config=bnb, dtype=DT, device_map={"": 0}, attn_implementation=a.attn)
    m.config.pad_token_id = PAD
    return m

@torch.no_grad()
def predict(model, items, bs=None):
    """Swap-TTA: average p(A,B,tie) with p(B,A,tie) mapped back. Returns probs (N,3)."""
    bs = bs or a.pred_bs
    model.eval(); order = np.argsort([len(i["x"]) for i in items]); out = np.zeros((len(items), 3), dtype=np.float64)
    for s in range(0, len(items), bs):
        idx = order[s:s + bs]; batch = [{k: v for k, v in items[j].items() if k != "y"} for j in idx]
        ps = []
        for key in ("x", "xs"):
            b = {k: v.cuda() for k, v in collate(batch, key).items()}
            with torch.autocast("cuda", dtype=DT):
                lg = model(**b).logits.float()
            ps.append(torch.softmax(lg, -1).cpu())
        out[idx] = ((ps[0] + ps[1][:, SWAP]) / 2).numpy()
    return out

def logloss(p, y):
    p = np.clip(p, 1e-15, 1); p /= p.sum(1, keepdims=True)
    return float(-np.log(p[np.arange(len(y)), y]).mean())

# ---------- run ----------
tr = pd.read_csv(f"{a.data}/train.csv")
te = pd.read_csv(f"{a.data}/test.csv")
tr = tr.sample(frac=1, random_state=a.seed).reset_index(drop=True)
nv = max(int(len(tr) * a.val_frac), 64) if not a.smoke else 64
va_df, tr_df = tr.iloc[:nv], tr.iloc[nv:]
if a.n_train: tr_df = tr_df.iloc[:a.n_train]
print(f"train {len(tr_df)}  val {len(va_df)}  test {len(te)}", flush=True)

t0 = time.time()
va = build(va_df, a.max_len); tei = build(te, a.max_len, with_label=False)
print("val token len mean/max:", np.mean([len(i['x']) for i in va]), max(len(i['x']) for i in va), flush=True)

model = load_base()
if a.predict_only:
    model = PeftModel.from_pretrained(model, f"{a.out}/adapter")
else:
    tri = build(tr_df, a.max_len)
    print(f"tokenized in {time.time()-t0:.0f}s", flush=True)
    if a.no_4bit:
        if a.grad_ckpt: model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=a.grad_ckpt)
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="SEQ_CLS",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for n, q in model.named_parameters():  # keep trainable params (LoRA + score head) in fp32
        if q.requires_grad:
            q.data = q.data.float()
            if "score" in n: q.data.zero_()  # zero-init head: start at uniform probs (log loss 1.0986)
    model.print_trainable_parameters()

    named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
    head_p = [q for n, q in named if "score" in n]; lora_p = [q for n, q in named if "score" not in n]
    params = head_p + lora_p
    print(f"head params {sum(q.numel() for q in head_p)}  lora params {sum(q.numel() for q in lora_p)}", flush=True)
    opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr}, {"params": head_p, "lr": a.lr * a.head_lr_mult}], weight_decay=0.01)
    steps_per_epoch = len(tri) // (a.bs * a.accum)
    total = max(1, int(steps_per_epoch * a.epochs)); warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=(DT == torch.float16))  # bf16 needs no loss scaling
    print(f"optimizer steps: {total}  (eff. batch {a.bs*a.accum})", flush=True)

    step, micro, run, t1 = 0, 0, [], time.time()
    model.train()
    def length_grouped_batches():
        """Shuffle, sort within mega-batches by length (less padding), then shuffle the batches."""
        perm = np.random.permutation(len(tri)); mega = a.bs * 64; out = []
        for m in range(0, len(perm), mega):
            chunk = sorted(perm[m:m + mega], key=lambda j: len(tri[j]["x"]))
            out += [chunk[k:k + a.bs] for k in range(0, len(chunk) - a.bs + 1, a.bs)]
        random.shuffle(out); return out

    while step < total:
        for bidx in length_grouped_batches():
            batch = [dict(tri[j]) for j in bidx]
            for b in batch:  # random A/B swap augmentation
                if random.random() < 0.5:
                    b["x"] = b["xs"]; b["y"] = int(SWAP[b["y"]])
            b = {k: v.cuda() for k, v in collate(batch).items()}
            with torch.autocast("cuda", dtype=DT):
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, b["labels"])
            scaler.scale(loss / a.accum).backward(); run.append(loss.item()); micro += 1
            if micro % a.accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if step % 10 == 0 or step == total:
                    el = time.time() - t1
                    print(f"step {step}/{total} loss {np.mean(run[-10*a.accum:]):.4f} lr {sched.get_last_lr()[0]:.2e} "
                          f"{el/step:.1f}s/step eta {(total-step)*el/step/60:.0f}m", flush=True)
                if step >= total: break
    model.save_pretrained(f"{a.out}/adapter")

pv = predict(model, va)
yv = np.array([i["y"] for i in va])
print(f"VAL log loss (swap-TTA): {logloss(pv.copy(), yv):.4f}   (uniform = 1.0986)", flush=True)
np.save(f"{a.out}/val_probs.npy", pv)

pt = predict(model, tei)
sub = pd.DataFrame({"id": te["id"], "winner_model_a": pt[:, 0], "winner_model_b": pt[:, 1], "winner_tie": pt[:, 2]})
sub.to_csv(f"{a.out}/submission_falcon.csv", index=False)
print("wrote", f"{a.out}/submission_falcon.csv")
