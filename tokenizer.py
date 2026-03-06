import re
import sys

# -----------------------------------------------------------
# 1) MODE 1 : découpe naïve par espaces
# -----------------------------------------------------------

def tokenize_whitespace(text):
    """
    Retourne une liste de tokens découpés par espaces.
    """
    return text.split()


# -----------------------------------------------------------
# 2) MODE 2 : découpe avec regex
# -----------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+|\d+|[.,;:!?()\"]")

def tokenize_regex(text):
    """
    Retourne une liste de tokens basés sur la regex.
    """
    return TOKEN_RE.findall(text)
# -----------------------------------------------------------
# 3) Interface en ligne de commande (CLI)
# -----------------------------------------------------------

if __name__ == "__main__":

    # 1) récupérer le mode
    mode = sys.argv[1]

    # 2) récupérer le texte
    text = " ".join(sys.argv[2:])

    # 3) choisir la fonction
    if mode == "whitespace":
        tokens = tokenize_whitespace(text)

    elif mode == "regex":
        tokens = tokenize_regex(text)

    else:
        print("Mode inconnu")
        sys.exit(1)

    # 4) afficher les tokens
    for token in tokens:
        print(token)
