import io
import os
import re
from pdfminer.high_level import extract_text
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Carregamento seguro das stopwords em português
# ============================================================
try:
    STOPWORDS = set(stopwords.words('portuguese'))
except Exception:
    # Faz download somente se necessário
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('portuguese'))

# Stemmer para reduzir palavras à raiz (stemming)
STEMMER = SnowballStemmer('portuguese')

# Caminho temporário para salvar PDFs antes de ler
TMP_PDF = (
    "/tmp/temp_upload.pdf"
    if os.name != "nt"
    else os.path.join(os.getenv("TEMP", "C:\\Windows\\Temp"), "temp_upload.pdf")
)

# Função: Extrair texto de PDF
# ============================================================
def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    """
    Extrai texto de um PDF usando pdfminer.

    Args:
        file_stream (BytesIO): Arquivo PDF enviado pelo usuário.

    Returns:
        str: Texto extraído do PDF.
    """
    # Garante que a leitura começa do início
    file_stream.seek(0)

    # Salva temporariamente o PDF para que pdfminer consiga processar
    with open(TMP_PDF, "wb") as f:
        f.write(file_stream.read())

    try:
        text = extract_text(TMP_PDF)
    except Exception:
        text = ""

    return text

# Função: Limpeza e Tokenização avançada do texto
# ============================================================
def clean_and_tokenize(text: str) -> str:
    """
    Limpa o texto removendo URLs, números, emails, stopwords
    e retorna versões simplificadas (stemming) das palavras.

    Esta função é usada pelo modelo local para melhorar
    a qualidade do treino e das previsões.

    Args:
        text (str): Texto original do email.

    Returns:
        str: Texto transformado e pronto para o modelo local.
    """

    # 1. Lowercase
    text = text.lower()

    # 2. Remover URLs, emails e menções
    text = re.sub(r"http\S+|www\S+|@\S+|\S+@\S+", " ", text)

    # 3. Remover números
    text = re.sub(r"\d+", " ", text)

    # 4. Remover excesso de espaços
    text = re.sub(r"\s+", " ", text)

    # 5. Tokenização simples (apenas letras)
    tokens = re.findall(r"\b[^\d\W]+\b", text, flags=re.UNICODE)

    # 6. Remover stopwords e palavras muito curtas
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    # 7. Stemming para reduzir variação de palavras
    stems = [STEMMER.stem(t) for t in tokens]

    # 8. Retorna texto pronto para TF-IDF
    return " ".join(stems)
