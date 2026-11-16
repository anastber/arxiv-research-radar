# One container for the demo: the FastAPI backend on localhost and the Streamlit frontend on
# $PORT (7860, the Hugging Face Spaces default). The paper index is not baked in: the API
# builds it from arXiv on startup and refreshes it daily (api/refresh.py).
FROM python:3.12-slim

# CPU-only torch: the default wheel bundles CUDA (several GB) that the demo never uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /home/user/app

# Download the embedding model at build time, so startup doesn't depend on the model hub.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .
EXPOSE 7860
CMD ["sh", "start.sh"]
