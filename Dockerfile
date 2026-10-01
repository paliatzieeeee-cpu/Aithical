# AI-thical — image για Google Cloud Run (μόνο CPU)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# Το pip της βασικής εικόνας είναι παλιό και δεν διαβάζει σωστά τα ονόματα
# πακέτων του ευρετηρίου του PyTorch (π.χ. typing_extensions), οπότε αναβαθμίζεται.
# PyTorch μόνο για CPU (χωρίς CUDA), στην ίδια έκδοση με το venv· η ρητή
# έκδοση «+cpu» αποκλείει την έκδοση CUDA του PyPI, απ' όπου έρχονται μόνο
# οι εξαρτήσεις του.
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
                --extra-index-url https://pypi.org/simple \
                torch==2.14.0+cpu

COPY requirements-cloudrun.txt .
RUN pip install -r requirements-cloudrun.txt

# Ο πολυγλωσσικός encoder κατεβαίνει ΚΑΤΑ ΤΟ BUILD, όχι σε κάθε εκκίνηση.
# Κρατάμε μόνο τα αρχεία που χρειάζεται το sentence-transformers (χωρίς τις
# εκδόσεις ONNX/OpenVINO/TF του ίδιου μοντέλου), και ελέγχουμε ότι φορτώνει.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', \
    ignore_patterns=['onnx/*', 'openvino/*', '*.h5', '*.msgpack', '*.ot', 'pytorch_model.bin']); \
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
print('encoder ok', m.encode(['test']).shape)"

# Στην εκτέλεση δεν επιτρέπεται λήψη από το Hugging Face
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY . .

# Εκτέλεση ως μη διαχειριστής· ο φάκελος data/ πρέπει να είναι εγγράψιμος
# για την καταγραφή χρήσης (usage_log.jsonl, προσωρινό στο Cloud Run)
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

# 1 worker (ένα αντίγραφο των μοντέλων στη μνήμη), 4 threads, timeout 120 s
CMD exec gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-8080} app:app
