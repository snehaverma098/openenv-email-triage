FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Assuming openenv uses uvicorn / fastapi and exposes on 8000. HF Space runs on 7860 by default for Gradio/FastAPI, but you can configure it.
# Actually, let's use uv run server or uvicorn
# The prompt mentions "uv run server"
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
EXPOSE 7860
