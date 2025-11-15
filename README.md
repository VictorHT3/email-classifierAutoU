📧 Email Classifier — Classificação Inteligente de E-mails

Aplicação completa para classificação automática de e-mails utilizando:

Modelo local (Machine Learning — TF-IDF + Regressão Logística)

Integração com OpenAI GPT-4o-mini para classificação temática e geração de resposta automática

Backend em FastAPI

Interface web simples em HTML

Suporte a upload de texto e PDF

O objetivo é identificar se um e-mail é Produtivo ou Improdutivo, classificar o tema principal e gerar uma resposta automática profissional.

📁 Estrutura do Projeto
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

🚀 Como Rodar Localmente
1. Instale as dependências
pip install -r requirements.txt

2. Adicione sua chave OpenAI

Crie um arquivo .env na raiz:

OPENAI_API_KEY=sua_chave_aqui

3. (Opcional) Treine o modelo local
python train/train_model.py


Isso gera:

app/model/classifier.pkl

4. Inicie a aplicação
uvicorn app.main:app --reload


Acesse no navegador:

👉 http://localhost:8000/

🧠 Funcionamento Técnico
🔹 Pré-processamento (nlp_utils.py)

Limpeza de texto (URLs, números, stopwords, emails)

Tokenização

Stemming (SnowballStemmer — PT-BR)

Preparação para o modelo local

🔹 Classificação Local (Machine Learning)

Usa:

TF-IDF com stopwords do NLTK

Regressão Logística

Arquivo de treino: train_model.py

🔹 Classificação via LLM (OpenAI)

O texto original é enviado para o modelo gpt-4o-mini, que retorna:

categoria: <CATEGORIA> | confianca: <1-10>

🔹 Geração de Resposta Automática

O LLM também cria uma resposta profissional baseada no texto original e categoria detectada.

📄 Dataset de Exemplo

O arquivo sample_data.csv segue formato:

text,label
"Preciso de confirmação do relatório.",Produtivo
"Bom dia, feliz natal!",Improdutivo


Treine novamente usando:

python train/train_model.py

🌐 Deploy

Pode ser hospedado em:

Render

Railway

Hugging Face Spaces

Azure / AWS / GCP

Replit

Comando recomendado:

uvicorn app.main:app --host 0.0.0.0 --port 80

🛠 Tecnologias

Python 3.10+

FastAPI

scikit-learn

NLTK

pdfminer.six

OpenAI API

HTML + CSS

📬 Contato

Desenvolvido por Victor Hugo Teixeira
Email: mrvictor2409@gmail.com

LinkedIn: https://www.linkedin.com/in/victorteixeira1b82b0161/
