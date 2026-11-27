# CPU image that serves the web app and API with GPT-2 small and Pythia-70m (and their SAEs) baked in.
# Works as-is on Hugging Face Spaces (Docker SDK, port 7860), Fly.io, Cloud Run, Render.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/hf PRISM_CACHE=/app/.cache/prism PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY prism ./prism
RUN pip install .
# Bake the default model and SAE layers into the image so cold starts need no downloads.
RUN python -c "from prism.workspace import Workspace; ws=Workspace(max_models=3); [(ws.model(n), [ws.sae(n, l) for l in ws.default_sae_layers(n)]) for n in ('gpt2', 'EleutherAI/pythia-70m-deduped')]"
COPY web ./web
RUN useradd -m -u 1000 app && chown -R app /app
USER app
EXPOSE 7860
CMD ["uvicorn", "prism.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
