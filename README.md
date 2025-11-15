# 📧 Email Classifier --- Classificação Inteligente de E-mails- Victor Teixeira

Aplicação completa para **classificação automática de e-mails**,
utilizando modelos de Machine Learning e LLM para identificar:

-   Se o e-mail é **Produtivo** ou **Improdutivo**
-   O **tema principal** da mensagem
-   Uma **resposta automática profissional** gerada pelo modelo

Inclui:

-   Modelo local (TF-IDF + Regressão Logística)
-   Integração com OpenAI `gpt-4o-mini` para classificação temática
    avançada
-   Backend em **FastAPI**
-   Interface web em **HTML**
-   Suporte a upload de **texto** e **PDF**

## 📁 Estrutura do Projeto

    project/
    │
    ├── app/
    │   ├── main.py                 # Backend FastAPI
    │   ├── model_utils.py          # Modelo local + LLM
    │   ├── nlp_utils.py            # Limpeza e pré-processamento
    │   ├── model/
    │   │   └── classifier.pkl      # Modelo local treinado
    │   ├── templates/
    │   │   └── index.html          # Interface web
    │   └── static/
    │       └── styles.css          # Estilos
    │
    ├── train/
    │   ├── train_model.py          # Treinamento do modelo local
    │   └── sample_data.csv         # Dataset de treino
    │
    ├── requirements.txt
    └── README.md

## 🚀 Como Rodar Localmente

### **1. Instale as dependências**

``` bash
pip install -r requirements.txt
```

### **2. Adicione sua chave OpenAI**

Crie um arquivo `.env` na raiz:

    OPENAI_API_KEY=sua_chave_aqui

### **3. (Opcional) Treine o modelo local**

``` bash
python train/train_model.py
```

### **4. Inicie a aplicação**

``` bash
uvicorn app.main:app --reload
```

Acesse no navegador:\
👉 **http://localhost:8000/**

## 🧠 Funcionamento Técnico

### 🔹 Pré-processamento (nlp_utils.py)

-   Limpeza de texto\
-   Remoção de URLs, números, e-mails e stopwords\
-   Tokenização\
-   Stemming (`SnowballStemmer` --- PT-BR)

### 🔹 Classificação Local (Machine Learning)

-   TF-IDF\
-   Regressão Logística\
-   Pipeline em `train/train_model.py`

### 🔹 Classificação via LLM (OpenAI)

Retorno esperado:

    categoria: <CATEGORIA> | confianca: <1-10>

### 🔹 Geração de Resposta Automática

Criação de resposta profissional com base no texto + categoria
detectada.

## 📄 Dataset de Exemplo

    text,label
    "Preciso de confirmação do relatório.",Produtivo
    "Bom dia, feliz natal!",Improdutivo

## 🌐 Deploy

Plataformas suportadas:

-   Render\
-   Railway\
-   Hugging Face Spaces\
-   Azure / AWS / GCP\
-   Replit

Comando recomendado:

``` bash
uvicorn app.main:app --host 0.0.0.0 --port 80
```

## 🛠 Tecnologias

-   Python 3.10+
-   FastAPI\
-   scikit-learn\
-   NLTK\
-   pdfminer.six\
-   OpenAI API\
-   HTML + CSS

## 📬 Contato

Desenvolvido por **Victor Hugo Teixeira**\
📧 Email: **mrvictor2409@gmail.com**\
🔗 LinkedIn: **https://www.linkedin.com/in/victorteixeira1b82b0161/**
