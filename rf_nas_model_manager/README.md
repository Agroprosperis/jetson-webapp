# Roboflow NAS Model Manager

A small UI-first utility for discovering trained Roboflow models, confirming RF-DETR NAS packages, optionally compiling them to TensorRT, and downloading ZIP files for Tilletia.

## Run

The manager image extends `tilletia-app:latest`, so TensorRT, CUDA-facing runtime packages, and their versions are identical to the Tilletia application. The Roboflow downloader runs in an isolated in-container Python environment because current `inference` requires NumPy 2 while Tilletia's OpenCV runtime uses the NumPy 1 ABI. Build that base image first, then start the manager:

```sh
docker compose -f deploy/docker-compose.yml build tilletia-app
docker compose -f rf_nas_model_manager/docker-compose.yml up --build
```

Open `http://127.0.0.1:8080`.

The Compose volume stores imported models and generated ZIPs under `rf_nas_model_manager/data/`. The Roboflow API key is supplied only in the browser and is not stored by the application.

## Workflow

1. Paste the private API key for a Roboflow workspace and connect.
2. Select a project, then a trained model version.
3. Click **Download and process**. The backend downloads the selected package and accepts it only when `model_config.json` identifies an `rfdetr` model with an ONNX backend and a supported detection task.
4. Optionally compile the imported ONNX model to an FP16 TensorRT engine.
5. Download the ZIP and upload it as an RF model on Tilletia's Models page.

An uncompiled ZIP contains the same source artifacts as the existing `prepare_deployment_package.py` flow. A compiled ZIP also contains the engine and the runtime sidecars Tilletia needs to load it immediately.

## Source layout

- `frontend/` contains the complete page markup, styling, and browser behavior.
- `backend/` contains the strict JSON API and model processing.
- `Dockerfile` and `docker-compose.yml` run both parts in one Flask container.

## Verification

The repository tests mock Roboflow and TensorRT. In accordance with the repository instructions, they are provided for CI or an operator to run and were not executed while implementing this app.
