import os
import joblib
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ================================
# Load local model
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "classifier.pkl")

print("🔍 Tentando carregar modelo em:", MODEL_PATH)

_local_model = None

try:
    _local_model = joblib.load(MODEL_PATH)
    print("Local model loaded:", MODEL_PATH)
except Exception as e:
    print("Local model not loaded:", e)

# ================================
# OpenAI client (new API)
# ================================
_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("OpenAI client not available:", e)

# ================================
# Local model classification
# ================================
def classify_local(text: str) -> Tuple[str, float]:
    if not _local_model:
        return None, 0.0

    try:
        proba = _local_model.predict_proba([text])[0]
        idx = proba.argmax()
        return _local_model.classes_[idx], float(proba[idx])
    except Exception as e:
        print("classify_local error:", e)
        return None, 0.0

# ================================
# LLM classification
# ================================
def classify_with_llm(text: str):
    """
    Classifica o email usando GPT-4o-mini e retorna:
    - categoria (str)
    - confiança de 0 a 10 (float)
    """

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classifique o email em uma CATEGORIA ÚNICA. "
                        "Responda SOMENTE no formato:\n"
                        "categoria: <CATEGORIA> | confianca: <1-10>\n\n"
                        "Exemplos de categorias: SUPORTE, COBRANÇA, PEDIDO, ELOGIO, SPAM, OUTROS."
                    )
                },
                {"role": "user", "content": text}
            ],
            temperature=0
        )

        # Conteúdo retornado
        content = resp.choices[0].message.content.strip()

        # -------------------------------
        # Normalização para parsing
        # -------------------------------
        content = content.replace("\n", " ").lower()

        categoria = "UNKNOWN"
        confianca = 0.0

        # -------------------------------
        # Extrai categoria e confiança
        # -------------------------------
        if "|" in content:
            partes = content.split("|")

            # Categoria
            if "categoria" in partes[0]:
                categoria = partes[0].split(":")[1].strip().upper()

            # Confiança
            if "confianca" in partes[1]:
                try:
                    confianca = float(partes[1].split(":")[1].strip())
                except:
                    confianca = 0.0

        else:
            # fallback — modelo às vezes responde só com o nome da categoria
            categoria = content.upper()

        return categoria, confianca

    except Exception as e:
        print("Erro no classify_with_llm:", e)
        return "ERROR", 0.0

# ================================
# LLM Automatic Response Generator
# ================================
def generate_response_with_llm(original_text: str, label: str) -> str:
    prompt = f"""
    Você é um assistente que gera respostas profissionais para emails.
    Categoria detectada: {label}
    Texto original:
    {original_text}

    Gere uma resposta clara, educada e objetiva.
    """

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um assistente especializado em responder emails."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return resp.choices[0].message.content.strip()