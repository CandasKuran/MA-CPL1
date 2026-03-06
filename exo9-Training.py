import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -------------------------------------------------
# 1) Dataset : lit mail + label, tokenize, renvoie tensors
# -------------------------------------------------
class MailDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.mails = df["mail"].astype(str).tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.mails)

    def __getitem__(self, idx):
        text = self.mails[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def label_to_int(x: str) -> int:
    x = str(x).strip()
    if x == "Answer":
        return 1
    if x == "NoAnswer":
        return 0
    raise ValueError(f"Label inconnu: {x} (attendu: Answer ou NoAnswer)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV avec colonnes: mail;label (label=Answer/NoAnswer)")
    parser.add_argument("--model_name", default="camembert-base", help="Modèle de base")
    parser.add_argument("--out_dir", default="./model_mail_classifier", help="Dossier de sortie")
    parser.add_argument("--epochs", type=int, default=5, help="Nombre d'epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Longueur max tokens")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -------------------------------------------------
    # 2) Lire CSV
    # -------------------------------------------------
    df = pd.read_csv(args.csv, sep=";", encoding="cp1252")

    df = df.dropna(subset=["mail", "label"]).copy()
    df["label"] = df["label"].apply(label_to_int)

    print(f"Nombre d'exemples: {len(df)}")

    # -------------------------------------------------
    # 3) Tokenizer + Modèle
    # -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    # -------------------------------------------------
    # 4) Dataset + DataLoader
    # -------------------------------------------------
    dataset = MailDataset(df, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    # -------------------------------------------------
    # 5) Optimiseur
    # -------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # -------------------------------------------------
    # 6) Entraînement
    # -------------------------------------------------
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(loader)
        acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.2f} | Accuracy (train): {acc:.2f}")

    # -------------------------------------------------
    # 7) Sauvegarde
    # -------------------------------------------------
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()