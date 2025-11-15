"""
train_model.py
--------------
Script responsável por treinar o modelo de classificação de emails
(Produtivo vs Improdutivo) utilizando TF-IDF + Regressão Logística.

Este script:
- Carrega o dataset CSV
- Faz validações
- Cria o pipeline de ML
- Treina o modelo
- Salva o modelo em disco

Autor: Victor Hugo Teixeira
"""

import os
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURAÇÕES E PATHS
# ============================================================

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "sample_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "app", "model", "classifier.pkl")

# Garante que o diretório do modelo existe
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_stopwords():
    """
    Carrega stopwords da NLTK em português.

    Returns:
        list: lista de stopwords.
    """
    nltk.download("stopwords", quiet=True)
    return stopwords.words("portuguese")


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Carrega o arquivo CSV contendo as amostras de treino.

    Args:
        csv_path (str): caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: dataset carregado.

    Raises:
        FileNotFoundError: se o arquivo não for encontrado.
        ValueError: se as colunas esperadas não existirem.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado em: {csv_path}")

    print(f"📄 Lendo arquivo de dados: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("O CSV precisa conter as colunas: 'text' e 'label'")

    return df


def build_pipeline(stop_words: list) -> Pipeline:
    """
    Cria o pipeline de processamento de texto + modelo.

    Args:
        stop_words (list): lista de stopwords em português.

    Returns:
        Pipeline: pipeline configurado.
    """
    print("⚙️ Criando pipeline TF-IDF + Regressão Logística...")

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=1_000_000,
            stop_words=stop_words
        )),
        ("clf", LogisticRegression(max_iter=300))
    ])


def train_and_save_model(pipeline: Pipeline, X, y, output_path: str):
    """
    Treina o modelo e salva em disco.

    Args:
        pipeline (Pipeline): pipeline de ML.
        X (pd.Series): textos de entrada.
        y (pd.Series): labels.
        output_path (str): onde salvar o modelo treinado.
    """
    print("🏋️ Treinando modelo...")
    pipeline.fit(X, y)

    print(f"💾 Salvando modelo em: {output_path}")
    joblib.dump(pipeline, output_path)

    print("✔ Modelo treinado e salvo com sucesso!")


# ============================================================
# MAIN
# ============================================================

def main():
    """Função principal — orquestra todo o processo de treinamento."""
    try:
        stop_words = load_stopwords()
        df = load_dataset(DATA_PATH)

        X = df["text"].astype(str)
        y = df["label"].astype(str)

        pipeline = build_pipeline(stop_words)
        train_and_save_model(pipeline, X, y, MODEL_PATH)

    except Exception as e:
        print("\n❌ ERRO DURANTE O TREINAMENTO DO MODELO:")
        print(str(e))


if __name__ == "__main__":
    main()