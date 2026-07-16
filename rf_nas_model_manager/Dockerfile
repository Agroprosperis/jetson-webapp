FROM tilletia-app:latest

WORKDIR /app

COPY rf_nas_model_manager/requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/rf-inference \
    && /opt/rf-inference/bin/python -m pip install --no-cache-dir "inference==1.3.5"

COPY rf_nas_model_manager /app/rf_nas_model_manager

RUN mkdir -p /data/models /data/packages /data/work

EXPOSE 8080

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:8080", "rf_nas_model_manager.backend.app:create_app()"]
