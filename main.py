
import os
import sys
import traceback
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, f1_score

# -------------------------
# Config (safe for CPU)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CANDIDATE_FILES = ["cleaned_requirements.csv", "semiLabelledData.csv",
                   "cleaned_requirements.CSV", "semiLabelledData.CSV"]
BERT_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 1   # keep low for safety; increase if GPU available

# -------------------------
# Utilities
# -------------------------
def find_data_file():
    if not os.path.isdir(DATA_DIR):
        return None, None
    files = os.listdir(DATA_DIR)
    # case-insensitive match to candidate names
    for cand in CANDIDATE_FILES:
        for f in files:
            if f.lower() == cand.lower():
                return os.path.join(DATA_DIR, f), f
    # fallback: look for any csv in folder
    for f in files:
        if f.lower().endswith(".csv"):
            return os.path.join(DATA_DIR, f), f
    return None, None

def try_read_csv(path):
    # try a couple encodings/engines
    for enc in ("utf-8", "latin1", "cp1252"):
        for engine in ("c", "python"):
            try:
                df = pd.read_csv(path, encoding=enc, engine=engine, low_memory=False)
                print(f"Read CSV ok with encoding={enc}, engine={engine}, shape={df.shape}")
                return df
            except Exception as e:
                # keep trying
                pass
    # last attempt: let pandas autodetect but show error
    return pd.read_csv(path, low_memory=False)

# Try to locate columns for req1, req2, label
def infer_columns(df):
    cols = [c.lower() for c in df.columns]
    # possible names:
    req1_candidates = ["req1","requirement1","requirement_1","r1","text1","sentence1","statement1"]
    req2_candidates = ["req2","requirement2","requirement_2","r2","text2","sentence2","statement2"]
    label_candidates = ["label","labels","binaryclass","multiclass","class","rel","relation"]

    req1_col = next((df.columns[i] for i,c in enumerate(cols) if c in req1_candidates), None)
    req2_col = next((df.columns[i] for i,c in enumerate(cols) if c in req2_candidates), None)
    label_col = next((df.columns[i] for i,c in enumerate(cols) if c in label_candidates), None)

    # fallback heuristics: look for two text columns (object dtype)
    if not req1_col or not req2_col:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if len(text_cols) >= 2:
            req1_col, req2_col = text_cols[0], text_cols[1]
    return req1_col, req2_col, label_col

class ReqDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k,v in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long)
        return item

# -------------------------
# Main
# -------------------------
def main():
    try:
        print("Base dir:", BASE_DIR)
        print("Looking for data files in:", DATA_DIR)
        path, fname = find_data_file()
        if not path:
            raise FileNotFoundError(f"No CSV files found in {DATA_DIR}. Please put your CSV(s) there.")

        print("Using file:", fname)
        df = try_read_csv(path)
        print("Columns found:", df.columns.tolist()[:20])

        req1_col, req2_col, label_col = infer_columns(df)
        print("Inferred columns -> req1:", req1_col, " req2:", req2_col, " label:", label_col)

        if not req1_col or not req2_col:
            raise ValueError("Could not find two text columns for requirements. Please ensure at least two text columns exist.")

        if label_col is None:
            # If no explicit label column, try to use columns named 'BinaryClass' or 'MultiClass' (case-insensitive)
            for alt in ["BinaryClass","binaryclass","MultiClass","multiclass"]:
                if alt in df.columns:
                    label_col = alt
                    print("Using alternative label column:", label_col)
                    break

        if label_col is None:
            # If still none, try to create dummy labels (all neutral=0) to run a demo
            print("No label column detected. Creating dummy labels (all zeros).")
            df['label'] = 0
            label_col = 'label'

        # Keep only rows with non-null reqs
        df = df.dropna(subset=[req1_col, req2_col]).reset_index(drop=True)
        df[req1_col] = df[req1_col].astype(str)
        df[req2_col] = df[req2_col].astype(str)
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
        df = df.dropna(subset=[label_col]).reset_index(drop=True)
        df[label_col] = df[label_col].astype(int)
        print("Final dataset shape:", df.shape)

        # Build mapping for labels
        unique_labels = sorted(df[label_col].unique().tolist())
        print("Unique label values:", unique_labels)
        label_map = {}
        if len(unique_labels) == 2:
            label_map = {unique_labels[0]: "support", unique_labels[1]: "conflict"}
        else:
            # generic names for each numeric label
            label_map = {v: f"class_{v}" for v in unique_labels}
        print("Label map:", label_map)

        # Prepare sentence pairs
        pairs = list(zip(df[req1_col].tolist(), df[req2_col].tolist()))
        labels = df[label_col]

        # Train/test split
        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            pairs, labels, test_size=0.15, random_state=42, stratify=labels if len(unique_labels)>1 else None
        )
        print("Train/Val sizes:", len(train_pairs), len(val_pairs))

        # Tokenizer
        print("Loading tokenizer:", BERT_NAME)
        tokenizer = BertTokenizer.from_pretrained(BERT_NAME)

        def tokenize_pairs(pairs_list):
            left = [a for a,_ in pairs_list]
            right = [b for _,b in pairs_list]
            enc = tokenizer(left, right,
                            padding='max_length',
                            truncation=True,
                            max_length=MAX_LEN,
                            return_tensors='pt')
            return enc

        train_enc = tokenize_pairs(train_pairs)
        val_enc = tokenize_pairs(val_pairs)

        # Create Datasets / Loaders
        train_dataset = ReqDataset(train_enc, pd.Series(train_labels))
        val_dataset = ReqDataset(val_enc, pd.Series(val_labels))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Model
        num_labels = len(unique_labels)
        print("Initializing model with num_labels =", num_labels)
        model = BertForSequenceClassification.from_pretrained(BERT_NAME, num_labels=num_labels)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print("Using device:", device)

        optimizer = AdamW(model.parameters(), lr=LR)

        # Training loop
        model.train()
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                # move inputs to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS} finished. Avg loss: {running_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['labels'].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(out.logits, dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_true.extend(labels_batch.cpu().numpy().tolist())

        print("\nClassification report (val):")
        try:
            target_names = [label_map[k] for k in sorted(label_map.keys())]
        except Exception:
            target_names = None
        print(classification_report(all_true, all_preds, target_names=target_names, zero_division=0))
        print("Precision (weighted):", precision_score(all_true, all_preds, average='weighted', zero_division=0))
        print("F1 (weighted):", f1_score(all_true, all_preds, average='weighted', zero_division=0))

        # Interactive predict
        def predict_pair(a, b):
            enc = tokenizer(a, b, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
            for k in enc:
                enc[k] = enc[k].to(device)
            with torch.no_grad():
                out = model(**enc)
                pred = int(torch.argmax(out.logits, dim=1).cpu().numpy()[0])
            return label_map.get(pred, str(pred))

        print("\nReady for quick predictions. Type 'exit' to quit.")
        while True:
            a = input("Req1 (or 'exit'): ").strip()
            if a.lower()=='exit':
                break
            b = input("Req2: ").strip()
            print("Predicted relation:", predict_pair(a,b))

    except Exception as e:
        print("ERROR: An exception occurred. Full traceback below:\n")
        traceback.print_exc()
        print("\nIf you want, copy & paste the traceback here so I can interpret it for you.")
        sys.exit(1)

if __name__ == "__main__":
    main()
