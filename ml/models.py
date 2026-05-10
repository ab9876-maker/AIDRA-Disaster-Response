"""
Machine Learning Module
Models: kNN, Naive Bayes, MLP Perceptron
Task: Predict victim survival probability and area risk level
Uses synthetic training data seeded from domain knowledge.
"""

import math
import random
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

random.seed(0)
np.random.seed(0)


# ─── Synthetic Dataset Generator ─────────────────────────────────────────────

def generate_training_data(n_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Features: [severity (1-3), time_waiting (0-30), proximity_to_risk (0-1),
               medical_kits_available (0-1), rescue_team_available (0-1)]
    Label: survived (1) or not (0)
    """
    X, y = [], []
    for _ in range(n_samples):
        severity = random.randint(1, 3)
        time_waiting = random.uniform(0, 30)
        proximity_risk = random.uniform(0, 1)
        kits_available = random.uniform(0, 1)
        team_available = random.randint(0, 1)

        # Survival model — shifted threshold to get ~50% class balance
        survival_score = (
            -0.3 * severity
            - 0.025 * time_waiting
            - 0.15 * proximity_risk
            + 0.25 * kits_available
            + 0.15 * team_available
            + random.gauss(0, 0.12)
        )
        survived = 1 if survival_score > -0.45 else 0

        X.append([severity, time_waiting, proximity_risk, kits_available, team_available])
        y.append(survived)

    return np.array(X, dtype=float), np.array(y)


def generate_risk_data(n_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Features: [fire_proximity, time_since_quake, road_damage_index, wind_direction_risk]
    Label: high_risk (1) or low_risk (0)
    """
    X, y = [], []
    for _ in range(n_samples):
        fire_prox = random.uniform(0, 1)
        time_quake = random.uniform(0, 20)
        road_damage = random.uniform(0, 1)
        wind_risk = random.uniform(0, 1)

        risk_score = (
            0.4 * fire_prox
            + 0.01 * (20 - time_quake)
            + 0.3 * road_damage
            + 0.2 * wind_risk
            + random.gauss(0, 0.08)
        )
        high_risk = 1 if risk_score > 0.4 else 0
        X.append([fire_prox, time_quake, road_damage, wind_risk])
        y.append(high_risk)

    return np.array(X, dtype=float), np.array(y)


# ─── kNN Classifier ──────────────────────────────────────────────────────────

class KNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.name = f"kNN (k={k})"

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def _distance(self, a, b):
        return math.sqrt(sum((x - z) ** 2 for x, z in zip(a, b)))

    def predict_proba(self, x: np.ndarray) -> float:
        dists = [(self._distance(x, self.X_train[i]), self.y_train[i])
                 for i in range(len(self.X_train))]
        dists.sort(key=lambda d: d[0])
        k_nearest = dists[:self.k]
        return sum(label for _, label in k_nearest) / self.k

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([1 if self.predict_proba(x) >= 0.5 else 0 for x in X])

    def predict_single(self, x) -> Tuple[int, float]:
        prob = self.predict_proba(np.array(x))
        return (1 if prob >= 0.5 else 0), prob


# ─── Naive Bayes Classifier (Gaussian NB) ────────────────────────────────────

class NaiveBayesClassifier:
    def __init__(self):
        self.name = "Naive Bayes (Gaussian)"
        self.classes = None
        self.priors = {}
        self.means = {}
        self.stds = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            self.means[c] = X_c.mean(axis=0)
            self.stds[c] = X_c.std(axis=0) + 1e-9

    def _log_likelihood(self, x, c):
        mean, std = self.means[c], self.stds[c]
        log_ll = -0.5 * np.sum(((x - mean) / std) ** 2) - np.sum(np.log(std))
        return log_ll + math.log(self.priors[c])

    def predict_proba(self, x: np.ndarray) -> float:
        scores = {c: self._log_likelihood(x, c) for c in self.classes}
        total = sum(math.exp(v) for v in scores.values())
        return math.exp(scores[1]) / total if total > 0 else 0.5

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([max(self.classes, key=lambda c: self._log_likelihood(x, c))
                         for x in X])

    def predict_single(self, x) -> Tuple[int, float]:
        x = np.array(x)
        prob = self.predict_proba(x)
        return (1 if prob >= 0.5 else 0), prob


# ─── MLP / Perceptron ────────────────────────────────────────────────────────

class MLPClassifier:
    """Simple 2-layer MLP with sigmoid activations."""

    def __init__(self, hidden_size: int = 10, lr: float = 0.05, epochs: int = 200):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.name = f"MLP (hidden={hidden_size})"
        self.W1 = self.b1 = self.W2 = self.b2 = None

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -20, 20)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_in = X.shape[1]
        self.W1 = np.random.randn(n_in, self.hidden_size) * 0.1
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, 1) * 0.1
        self.b2 = np.zeros(1)
        y = y.reshape(-1, 1).astype(float)

        for _ in range(self.epochs):
            # Forward
            z1 = X @ self.W1 + self.b1
            a1 = self.sigmoid(z1)
            z2 = a1 @ self.W2 + self.b2
            a2 = self.sigmoid(z2)

            # Backward
            d2 = (a2 - y) * a2 * (1 - a2)
            d1 = (d2 @ self.W2.T) * a1 * (1 - a1)

            self.W2 -= self.lr * a1.T @ d2 / len(X)
            self.b2 -= self.lr * d2.mean(axis=0)
            self.W1 -= self.lr * X.T @ d1 / len(X)
            self.b1 -= self.lr * d1.mean(axis=0)

    def predict_proba(self, x: np.ndarray) -> float:
        x = np.array(x).reshape(1, -1)
        a1 = self.sigmoid(x @ self.W1 + self.b1)
        a2 = self.sigmoid(a1 @ self.W2 + self.b2)
        return float(a2[0][0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        a1 = self.sigmoid(X @ self.W1 + self.b1)
        a2 = self.sigmoid(a1 @ self.W2 + self.b2)
        return (a2.flatten() >= 0.5).astype(int)

    def predict_single(self, x) -> Tuple[int, float]:
        prob = self.predict_proba(x)
        return (1 if prob >= 0.5 else 0), prob


# ─── Evaluation Metrics ──────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ─── MLManager: trains and evaluates all models ──────────────────────────────

class MLManager:
    def __init__(self):
        self.survival_models = {}
        self.risk_models = {}
        self.survival_metrics = {}
        self.risk_metrics = {}
        self._train()

    def _split(self, X, y, ratio=0.8):
        n = int(len(X) * ratio)
        idx = np.random.permutation(len(X))
        return X[idx[:n]], X[idx[n:]], y[idx[:n]], y[idx[n:]]

    def _train(self):
        # Survival prediction models
        X_s, y_s = generate_training_data(500)
        X_str, X_ste, y_str, y_ste = self._split(X_s, y_s)

        knn_s = KNNClassifier(k=7)
        nb_s = NaiveBayesClassifier()
        mlp_s = MLPClassifier(hidden_size=12, lr=0.05, epochs=300)

        for m in [knn_s, nb_s, mlp_s]:
            m.fit(X_str, y_str)

        self.survival_models = {"kNN": knn_s, "Naive Bayes": nb_s, "MLP": mlp_s}
        for name, m in self.survival_models.items():
            y_pred = m.predict(X_ste)
            self.survival_metrics[name] = compute_metrics(y_ste, y_pred)
            self.survival_metrics[name]["model"] = name
            self.survival_metrics[name]["task"] = "survival"

        # Risk level models
        X_r, y_r = generate_risk_data(400)
        X_rtr, X_rte, y_rtr, y_rte = self._split(X_r, y_r)

        knn_r = KNNClassifier(k=5)
        nb_r = NaiveBayesClassifier()

        for m in [knn_r, nb_r]:
            m.fit(X_rtr, y_rtr)

        self.risk_models = {"kNN": knn_r, "Naive Bayes": nb_r}
        for name, m in self.risk_models.items():
            y_pred = m.predict(X_rte)
            self.risk_metrics[name] = compute_metrics(y_rte, y_pred)
            self.risk_metrics[name]["model"] = name
            self.risk_metrics[name]["task"] = "risk"

    def predict_survival(self, severity: int, time_waiting: float,
                          proximity_risk: float, kits: float, team: int) -> Dict:
        x = [severity, time_waiting, proximity_risk, kits, team]
        results = {}
        for name, m in self.survival_models.items():
            label, prob = m.predict_single(x)
            results[name] = {"survived": bool(label), "probability": round(prob, 3)}
        # Ensemble average
        avg_prob = sum(r["probability"] for r in results.values()) / len(results)
        results["ensemble"] = {"survived": avg_prob >= 0.5, "probability": round(avg_prob, 3)}
        return results

    def predict_risk(self, fire_prox: float, time_quake: float,
                     road_damage: float, wind_risk: float) -> Dict:
        x = [fire_prox, time_quake, road_damage, wind_risk]
        results = {}
        for name, m in self.risk_models.items():
            label, prob = m.predict_single(x)
            results[name] = {"high_risk": bool(label), "probability": round(prob, 3)}
        avg = sum(r["probability"] for r in results.values()) / len(results)
        results["ensemble"] = {"high_risk": avg >= 0.5, "probability": round(avg, 3)}
        return results
