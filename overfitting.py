import math
import random

# -------------------------
# Données (depuis la fiche)
# -------------------------
TRAIN = [
    # Règle normale (x1 > x2 → 1)
    (4.8, 1.0, 1),
    (4.2, 0.5, 1),
    (3.9, 1.1, 1),
    (3.5, 0.8, 1),
    (2.9, 1.0, 1),
    (4.5, 2.0, 1),
    (3.8, 2.5, 1),
    (2.5, 0.4, 1),
    (0.5, 4.5, 0),
    (1.0, 4.2, 0),
    (0.8, 3.9, 0),
    (1.5, 3.5, 0),
    (1.2, 2.9, 0),
    (2.0, 4.5, 0),
    (2.5, 3.8, 0),
    (0.4, 2.5, 0),
    # Bruit volontaire (mauvais labels)
    (4.6, 0.5, 0),  # devrait être 1
    (4.0, 1.0, 0),  # devrait être 1
    (3.7, 1.2, 0),  # devrait être 1
    (0.6, 4.7, 1),  # devrait être 0
    (1.0, 4.4, 1),  # devrait être 0
    (1.5, 3.9, 1),  # devrait être 0
]

VAL = [
    (4.9, 1.0, 1),
    (4.1, 0.7, 1),
    (3.6, 1.2, 1),
    (3.2, 0.9, 1),
    (4.4, 2.1, 1),
    (2.8, 1.5, 1),
    (3.9, 0.8, 1),
    (4.7, 2.0, 1),
    (3.3, 1.1, 1),
    (4.0, 1.9, 1),
    (0.7, 4.8, 0),
    (1.2, 4.1, 0),
    (0.9, 3.6, 0),
    (1.1, 3.2, 0),
    (2.0, 4.4, 0),
    (1.4, 2.8, 0),
    (0.8, 3.5, 0),
    (1.6, 4.0, 0),
    (2.2, 3.7, 0),
    (0.5, 2.6, 0),
]

ETA = 0.01  # learning rate
EPOCHS = 300


# -------------------------
# Modèle: 1 neurone sigmoid
# -------------------------
def sigmoid(z: float) -> float:
    # retourner 1 / (1 + exp(-z))
    return 1.0 / (1.0 + math.exp(-z))


def forward(x1: float, x2: float, w1: float, w2: float, b: float) -> float:
    """Retourne p = sigmoid(w1*x1 + w2*x2 + b)"""
    z = w1 * x1 + w2 * x2 + b
    return sigmoid(z)


def mse_loss(y: int, p: float) -> float:
    """
    Loss MSE pour classification binaire.
    L = 1/2 * (y - p)^2
    """
    return 0.5 * (y - p) ** 2


def predict(x1: float, x2: float, w1: float, w2: float, b: float) -> int:
    p = forward(x1, x2, w1, w2, b)
    return 1 if p >= 0.5 else 0


def accuracy(dataset, w1: float, w2: float, b: float) -> float:
    correct = 0
    for x1, x2, y_true in dataset:
        y_pred = predict(x1, x2, w1, w2, b)
        if y_pred == y_true:
            correct += 1
    return correct / len(dataset)


# -------------------------
# Entraînement (SGD)
# -------------------------
def train():
    # Init aléatoire (petit) des paramètres
    w1 = random.uniform(-0.5, 0.5)
    w2 = random.uniform(-0.5, 0.5)
    b = random.uniform(-0.5, 0.5)

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        # 1 epoch = passer sur tout TRAIN
        for x1, x2, y in TRAIN:
            # forward
            p = forward(x1, x2, w1, w2, b)

            # loss (pour affichage)
            total_loss += mse_loss(y, p)

            # gradient (MSE + sigmoid):
            # dL/dz = (p - y) * p*(1-p)
            dz = (p - y) * p * (1 - p)

            # gradients des paramètres
            dw1 = dz * x1
            dw2 = dz * x2
            db = dz

            # update
            w1 -= ETA * dw1
            w2 -= ETA * dw2
            b -= ETA * db

        train_acc = accuracy(TRAIN, w1, w2, b)
        val_acc = accuracy(VAL, w1, w2, b)

        # Affichage seulement tous les 25 epochs
        if epoch % 25 == 0 or epoch == 1 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {total_loss:.4f} | "
                f"Train: {train_acc*100:6.2f}% | "
                f"Val: {val_acc*100:6.2f}%"
            )

    print("\nParamètres finaux:")
    print(f"w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")


if __name__ == "__main__":
    train()
