import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# =============================
# PATHS
# =============================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "sample_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "app", "model", "classifier.pkl")

# Garante que o diretório exista
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print("📄 Lendo arquivo de dados:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Validação
if "text" not in df or "label" not in df:
    raise ValueError("O CSV precisa conter as colunas: 'text' e 'label'")

X = df["text"].astype(str)
y = df["label"].astype(str)

# =============================
# MODELO
# =============================
print("🛠 Criando pipeline TF-IDF + Regressão Logística...")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        stop_words="portuguese"
    )),
    ("clf", LogisticRegression(max_iter=200))
])

print("🏋️ Treinando modelo...")
pipeline.fit(X, y)

# =============================
# SALVAR MODELO
# =============================
print("💾 Salvando modelo em:", MODEL_PATH)
joblib.dump(pipeline, MODEL_PATH)

print("✔ Modelo treinado e salvo com sucesso!")