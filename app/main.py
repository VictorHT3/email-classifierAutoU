from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io

from .nlp_utils import extract_text_from_pdf, clean_and_tokenize
from .model import classify_local, classify_with_llm ,generate_response_with_llm

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# ===========================
# Home Page
# ===========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ===========================
# Prediction Endpoint
# ===========================
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    text_input: str = Form(None),
    file: UploadFile = File(None)
):
    raw_text = ""

    # -------------------------
    # Extract text from file
    # -------------------------
    if file and file.filename:
        content = await file.read()

        if file.filename.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(io.BytesIO(content))
        else:
            raw_text = content.decode(errors="ignore")

    # -------------------------
    # Direct text input
    # -------------------------
    elif text_input:
        raw_text = text_input

    # -------------------------
    # No text provided
    # -------------------------
    if not raw_text.strip():
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Nenhum texto detectado."}
        )

    # -------------------------
    # Clean text for local model
    # -------------------------
    cleaned = clean_and_tokenize(raw_text)

    # -------------------------
    # Classification (ONLY LOCAL MODEL)
    # -------------------------
    label_local, conf_local = classify_local(cleaned)
    categoria, categoria_conf = classify_with_llm(raw_text)

    if label_local is None:
        chosen_label = "Erro no modelo"
        confidence = 0
    else:
        chosen_label = label_local
        confidence = conf_local

    # -------------------------
    # Generate automatic response
    # -------------------------
    suggested_response = generate_response_with_llm(raw_text, chosen_label)

    # -------------------------
    # Return result to template
    # -------------------------
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "original": raw_text,
            "productivity": chosen_label,  # Produtivo / improdutivo
            "productivity_conf": round(conf_local, 3),
            "category": categoria,  # Categoria do assunto
            "category_conf": categoria_conf,
            "response_suggested": suggested_response,
            "confidence": round(confidence, 3),
        },
    )