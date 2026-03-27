import csv
import requests

API_URL = "http://10.229.43.154:11434/api/generate"

def analyse_message(texte):
    prompt = f"""
    Réponds STRICTEMENT par un seul mot : Oui ou Non.

    - Oui = nécessite une réponse
    - Non = ne nécessite pas de réponse

    Ne donne AUCUNE phrase, AUCUNE explication.

    Message : {texte}
    """

    r = requests.post(API_URL, json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })

    return r.json()["response"].strip()

with open("emailss.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        message = row["email_text"]
        attendu = row["label_attendu"]
        resultat = analyse_message(message)

        print("Message :", message)
        print("Réponse IA :", resultat)
        print("Attendu :", attendu)
        print("-" * 40)