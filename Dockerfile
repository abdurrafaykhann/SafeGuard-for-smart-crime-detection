FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY safeguard safeguard
COPY scripts scripts
COPY models models
COPY run.py .
RUN mkdir -p /app/instance && chown -R app /app
USER app
VOLUME /app/instance
EXPOSE 5000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/healthz')" || exit 1
CMD ["python", "run.py", "--host", "0.0.0.0"]
