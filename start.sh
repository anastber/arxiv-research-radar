#!/bin/sh
# The API listens on localhost only: visitors reach it through the Streamlit frontend, which
# is what lets the API trust the X-Visitor-IP header for per-visitor rate limiting.
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
exec streamlit run app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true
