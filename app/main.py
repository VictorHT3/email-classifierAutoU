"""
Aplicação FastAPI responsável por:

- Renderizar a interface web (index.html)
- Aceitar upload de arquivos ou texto manual
- Extrair texto de PDFs e arquivos texto
- Limpar o texto antes de enviar ao modelo local
- Classificar o conteúdo (local + LLM, quando existente)
- Gerar uma resposta sugerida com LLM
- Retornar o resultado para o template

Autor: Victor Hugo Teixeira
"""

import io
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .nlp_utils import extract_text_from_pdf, clean_and_tokenize
from .model import (
    classify_local,
    classify_with_llm,
    generate_response_with_llm
)


# Inicialização da Aplicação
# ============================================================

app = FastAPI(title="Email Classifier", version="1.0.0")

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Home Page
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Renderiza a página inicial da aplicação.
    Args:
    request (Request): objeto de requisição HTTP.
    Returns:
    TemplateResponse: página HTML inicial.
    """
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction Endpoint
# ============================================================

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    text_input: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Endpoint responsável por receber texto ou arquivo,
    processar o conteúdo e retornar a classificação.
    Fluxo:
        Recebe texto manual ou upload
        Extrai e limpa o texto
        Classifica via modelo local
        Classificação adicional via LLM (se ativo)
        Gera resposta automática
        Retorna resultado ao template
    Args:
        request (Request): Requisição HTTP.
        text_input (str): Entrada de texto manual fornecida pelo usuário.
        file (UploadFile): Arquivo enviado pelo usuário (PDF, txt etc).
    Returns:
        TemplateResponse: página com os resultados.
    """

    raw_text = ""

    # Extração do texto caso um arquivo seja enviado
    # --------------------------------------------------------
    if file and file.filename:
        try:
            file_content = await file.read()

            if file.filename.lower().endswith(".pdf"):
                raw_text = extract_text_from_pdf(io.BytesIO(file_content))
            else:
                # Assume arquivos .txt, .eml ou similares
                raw_text = file_content.decode(errors="ignore")

        except Exception as exc:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": f"Erro ao processar arquivo: {str(exc)}"
                }
            )

    # Caso o usuário tenha enviado texto diretamente
    # --------------------------------------------------------
    elif text_input:
        raw_text = text_input

    # Falha caso nenhum texto tenha sido enviado
    # --------------------------------------------------------
    if not raw_text.strip():
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Nenhum texto detectado."}
        )

    # Limpeza do texto — preparação para o modelo local
    # --------------------------------------------------------
    cleaned_text = clean_and_tokenize(raw_text)

    # Classificação usando apenas o modelo local
    # --------------------------------------------------------
    label_local, confidence_local = classify_local(cleaned_text)

    if label_local is None:
        chosen_label = "Erro no modelo"
        confidence_local = 0.0
    else:
        chosen_label = label_local

    # Classificação complementar realizada via LLM
    category_label, category_conf = classify_with_llm(raw_text)

    # Geração da resposta automática usando LLM
    # --------------------------------------------------------
    suggested_response = generate_response_with_llm(raw_text, chosen_label)

    # Retorno do resultado para a interface
    # --------------------------------------------------------
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "original": raw_text,
            "productivity": chosen_label,
            "productivity_conf": round(confidence_local, 3),
            "category": category_label,
            "category_conf": category_conf,
            "response_suggested": suggested_response,
            "confidence": round(confidence_local, 3)
        },
    )