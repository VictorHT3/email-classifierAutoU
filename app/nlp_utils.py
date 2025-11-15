import io
import os
import re
from pdfminer.high_level import extract_text
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# download stopwords if needed
try:
    STOPWORDS = set(stopwords.words('portuguese'))
except:
    import nltk as _nltk
    _nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('portuguese'))

STEMMER = SnowballStemmer('portuguese')

TMP_PDF = (
    "/tmp/temp_upload.pdf"
    if os.name != 'nt'
    else os.path.join(os.getenv('TEMP', 'C:\\Windows\\Temp'), 'temp_upload.pdf')
)


def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    file_stream.seek(0)
    with open(TMP_PDF, 'wb') as f:
        f.write(file_stream.read())

    try:
        text = extract_text(TMP_PDF)
    except Exception:
        text = ''

    return text


def clean_and_tokenize(text: str) -> str:
    text = text.lower()

    text = re.sub(r'http\S+|www\S+|@\S+|\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    tokens = re.findall(r"\b[^\d\W]+\b", text, flags=re.UNICODE)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    stems = [STEMMER.stem(t) for t in tokens]

    return ' '.join(stems)
