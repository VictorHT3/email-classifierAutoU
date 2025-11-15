import os
import joblib
from typing import Tuple
from dotenv import load_dotenv

# Carregar variáveis de ambiente
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Caminhos e carregamento do modelo local
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "classifier.pkl")

print(f"🔍 Tentando carregar modelo local em: {MODEL_PATH}")

_local_model = None
try:
    _local_model = joblib.load(MODEL_PATH)
    print(f"✅ Modelo local carregado com sucesso.")
except Exception as e:
    print(f" Falha ao carregar modelo local: {e}")

# Cliente OpenAI (novo formato)
# ==========================================
_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI

        _client = OpenAI(api_key=OPENAI_API_KEY)
        print("🔑 Cliente OpenAI inicializado.")
    except Exception as e:
        print("Erro ao inicializar cliente OpenAI:", e)
else:
    print("⚠️ Nenhuma OPENAI_API_KEY encontrada.")


# Função — Classificação com modelo local
# ==========================================
def classify_local(text: str) -> Tuple[str, float]:
    """
    Usa o modelo local (Logistic Regression + TF-IDF) para classificar o texto.
    Retorna:
        label (str): rótulo previsto
        confidence (float): probabilidade entre 0 e 1
    """
    if not _local_model:
        return None, 0.0

    try:
        # predict_proba retorna um array de probabilidades
        proba = _local_model.predict_proba([text])[0]
        idx = proba.argmax()

        return _local_model.classes_[idx], float(proba[idx])

    except Exception as e:
        print("Erro em classify_local:", e)
        return None, 0.0


# Função — Classificação com LLM
# ==========================================
def classify_with_llm(text: str):
    """
    Classifica o texto usando GPT-4o-mini.
    O LLM deve responder no formato:
        categoria: <CATEGORIA> | confianca: <1-10>
    Retorna:
        categoria (str)
        confianca (float)
    """

    if not _client:
        print("⚠️ LLM não disponível.")
        return "LLM_INDISPONIVEL", 0.0

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classifique o email em UMA ÚNICA categoria.\n"
                        "Formato obrigatório da resposta:\n"
                        "categoria: <CATEGORIA> | confianca: <1-10>\n"
                        "Categorias sugeridas: SUPORTE, COBRANÇA, PEDIDO, "
                        "ELOGIO, SPAM, OUTROS."
                    )
                },
                {"role": "user", "content": text}
            ],
        )

        content = resp.choices[0].message.content.strip().lower()
        content = content.replace("\n", " ")

        categoria = "UNKNOWN"
        confianca = 0.0

        # Processamento da string retornada
        if "|" in content:
            categoria_raw, conf_raw = content.split("|")

            if "categoria" in categoria_raw:
                categoria = categoria_raw.split(":")[1].strip().upper()

            if "confianca" in conf_raw:
                try:
                    confianca = float(conf_raw.split(":")[1].strip())
                except:
                    confianca = 0.0
        else:
            categoria = content.upper()

        return categoria, confianca

    except Exception as e:
        print(" Erro em classify_with_llm:", e)
        return "ERROR", 0.0

# Função — Gerador de Resposta Automática
# ==========================================
def generate_response_with_llm(original_text: str, label: str) -> str:
    """
    Gera uma resposta automática com base no texto original e na categoria prevista.
    Retorna:
        texto (str): resposta sugerida pelo LLM
    """

    if not _client:
        return "LLM não disponível para gerar resposta."

    prompt = f"""
    Você é um assistente profissional que escreve emails claros e educados.
    Categoria detectada: {label}

    Texto original do usuário:
    {original_text}

    Gere uma resposta objetiva, educada e direta.
    """

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em responder emails."},
                {"role": "user", "content": prompt}
            ]
        )

        return resp.choices[0].message.content.strip()

    except Exception as e:
        print("Erro em generate_response_with_llm:", e)
        return "Erro ao gerar resposta automática."
