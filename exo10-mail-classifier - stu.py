import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# 1. Fonctions de conversion des labels
# =========================
def label_to_int(x: str) -> int:
    x = str(x).strip()
    if x == "Answer":
        return 1
    elif x == "NoAnswer":
        return 0
    else:
        raise ValueError(f"Label inconnu: {x}")


def int_to_label(i: int) -> str:
    if i == 1:
        return "Answer"
    elif i == 0:
        return "NoAnswer"
    else:
        raise ValueError(f"Valeur entière inconnue: {i}")


def main():
    # =========================
    # 2. Chargement des arguments
    # =========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./model_mail_classifier", help="Dossier du modèle fine-tuné")
    parser.add_argument("--csv", help="CSV validation avec colonnes: mail;label")
    parser.add_argument("--text", help="Phrase unique à classifier")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--show_errors", type=int, default=10, help="Nombre d'erreurs à afficher")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # =========================
    # 3. Chargement du modèle et du tokenizer
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    # =========================
    # 4. Mode texte unique
    # =========================
    if args.text:
        with torch.no_grad():
            inputs = tokenizer(
                args.text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=args.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            y_pred = int(torch.argmax(probs).item())
            score = float(probs[y_pred].item())

            print(f"Prédiction: {int_to_label(y_pred)}")
            print(f"Score: {score:.3f}")
        return

    if not args.csv:
        raise ValueError("Vous devez fournir soit --csv, soit --text.")

    # =========================
    # 5. Lecture et préparation du CSV
    # =========================
    df = pd.read_csv(args.csv, sep=";")
    df.columns = df.columns.str.strip()

    if "mail" not in df.columns and "Mail" in df.columns:
        df = df.rename(columns={"Mail": "mail"})

    if not {"mail", "label"}.issubset(df.columns):
        raise ValueError(f"Colonnes attendues: mail;label. Colonnes trouvées: {list(df.columns)}")

    df = df.dropna(subset=["mail", "label"]).copy()
    df["y_true"] = df["label"].apply(label_to_int)

    total = 0

    # =========================
    # 6. Initialisation des métriques
    # =========================
    TN = FP = FN = TP = 0
    errors = []

    # =========================
    # 7. Boucle d’inférence
    # =========================
    with torch.no_grad():
        for _, row in df.iterrows():
            text = str(row["mail"])
            y_true = int(row["y_true"])

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=args.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            y_pred = int(torch.argmax(probs).item())
            score = float(probs[y_pred].item())

            total += 1

            if y_pred != y_true:
                errors.append((text, int_to_label(y_true), int_to_label(y_pred), score))

            if y_true == 0 and y_pred == 0:
                TN += 1
            elif y_true == 0 and y_pred == 1:
                FP += 1
            elif y_true == 1 and y_pred == 0:
                FN += 1
            elif y_true == 1 and y_pred == 1:
                TP += 1

    # =========================
    # 8. Calcul des métriques
    # =========================
    recall_answer = TP / (TP + FN) if (TP + FN) else 0.0

    print(f"\nValidation: {total} exemples")
    print(f"Recall (Answer): {recall_answer:.3f}")

    print("\nMatrice de confusion (lignes=vrai, colonnes=prédit)")
    print("                 Pred NoAnswer    Pred Answer")
    print(f"True NoAnswer        {TN:4d}          {FP:4d}")
    print(f"True Answer          {FN:4d}          {TP:4d}")

    # =========================
    # 9. Affichage des erreurs
    # =========================
    if errors:
        print(f"\nExemples d'erreurs (max {args.show_errors}) :")
        for i, (text, yt, yp, score) in enumerate(errors[: args.show_errors], start=1):
            short = (text[:140] + "…") if len(text) > 140 else text
            print(f"{i:02d}. Vrai={yt:<8} | Prédit={yp:<8} | score={score:.3f} | mail='{short}'")
    else:
        print("\nAucune erreur sur ce set.")


if __name__ == "__main__":
    main()