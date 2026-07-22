# Application overview
The dashboard is the main screen for selecting a camera or uploaded video, choosing a model, starting/stopping analysis, and taking snapshots. Use `Run` to start processing, `Stop` to stop it, `Take Snapshot` or `F4` to save a manual evidence frame, `See Results` to review saved analyses, and `Model Catalog` to upload, compile, select, or delete models. The results screen lists analysis folders, evidence images, videos, CSV files, per-class object counts, and S values.

# Setup step
REPO_DIR is the directory where the github repository is cloned
```
export REPO_DIR=<insert path to cloned repository here>
```

# Production autostart (systemd)
Use deployment scripts from `app/deploy` to install a system-wide service.

Install and start service (builds images and enables systemd unit):
```
cd ${REPO_DIR}/deploy
sudo ./install-autostart.sh
```

Optional: manual service control:
```
sudo systemctl status tilletia-app.service
sudo systemctl restart tilletia-app.service
sudo systemctl stop tilletia-app.service
```

Uninstall system service only (keeps all data and configs):
```
cd ${REPO_DIR}/deploy
sudo ./install-autostart.sh --uninstall
```

Optional: Logs:
```
sudo journalctl -u tilletia-app.service -f
```

After a reinstall or a clean data reset, the best default entry page is:
```
http://localhost:8000/login
```

Bootstrap local user:
```
admin / admin
```

# Authentication and RBAC
The application uses local authentication with bearer access tokens.

- `admin`: full access to the dashboard configuration, models page, users page, Swagger, and all results.
- `user`: can run and stop analysis, view REST status endpoints, and see only their own results.
- New users are created by an admin and must change the initial password on first login.
- Owner columns are shown only for admin views where ownership is relevant.

## Upload result images to Roboflow

An administrator can configure a private Roboflow API token and project ID on the Settings page. Before the token field is shown, the server reports whether secure persistent storage is usable. The token is never returned by the web API or rendered into a page. The project ID is non-secret and persists separately; it is the last part of the Roboflow project URL and is also the dataset name used by the upload API.

For the systemd deployment, startup checks for TPM 2.0 support and provisions a persistent TPM-encrypted 32-byte master-key credential at `/var/lib/tilletia-app/credentials/roboflow-master-key.cred`. On every service start, the same key is decrypted into the volatile `/run/tilletia/secrets` ramfs and mounted read-only into the container. The application uses that key with AES-256-GCM and stores only the nonce and authenticated ciphertext in SQLite. App reinstall and ordinary service or machine restarts preserve decryptability because the TPM-encrypted credential is retained. The credential remains TPM-bound but deliberately has no PCR binding, avoiding lockout during routine OS or firmware updates at the cost of weaker protection against alternate boot environments on the same machine.

If the TPM, systemd credential support, ramfs mount, provisioning, or decryption is unavailable, the admin sees a warning before entering the token. By default, the token is held only in the single application process and is lost on service restart. The admin may instead explicitly check the warning checkbox to persist it unencrypted in owner-only `/app/runs/roboflow.json` mode `0600`. This fallback is not encryption at rest.

An existing encrypted token is never silently downgraded if the TPM key later becomes unavailable. Restore the secure-storage key or enter a replacement token and choose memory or the confirmed plaintext fallback. Clearing or replacing the TPM can make its encrypted credential unrecoverable. Legacy plaintext settings are automatically migrated to encrypted storage when secure storage is available; without it, an unconfirmed legacy token is moved into memory and the plaintext file is removed.

On the Results page, administrators can select multiple image thumbnails. The **Upload to Roboflow** button appears when at least one image is selected and uploads the selection to the configured project's `train` split. Regular users do not see these controls and cannot call the Roboflow configuration or upload APIs.


Optional: Run stack without installing service:
```
cd ${REPO_DIR}/deploy
./start.sh
```
This mode builds the `tilletia-app:latest` image from the current tree, mounts models from `${REPO_DIR}/data/model`, and uses `${REPO_DIR}/data/output_hq` and `${REPO_DIR}/data/runs` for outputs.
Missing runtime folders under `data/` are created automatically by deploy scripts. Standalone mode intentionally uses an empty, nonpersistent runtime secrets directory, so the Settings page offers memory-only storage or the explicitly confirmed plaintext fallback.

# YOLO model deployment
YOLO model weights must be downloaded from the Roboflow site. Upload the downloaded `.pt` weights in the app from `Model Catalog`, then compile the uploaded model as TensorRT from the UI according to [Compile model as TensorRT from UI](#compile-model-as-tensorrt-from-ui). After compilation, select the generated engine on the dashboard and run analysis.

# Roboflow NAS model deployment
From `deploy` folder run:
```
sudo docker compose -f docker-compose.deploy-model.yml run --rm --build deploy-model <modelid> <ROBOFLOWAPI_KEY>
```
It creates <modelid>.zip archive which could be directly uploaded to the app

A UI-first version of this workflow is available in `rf_nas_model_manager/`. It discovers the workspace, projects, and trained versions from a browser-provided API key, verifies RF-DETR packages, optionally compiles TensorRT, and produces Tilletia-compatible ZIP files. See `rf_nas_model_manager/README.md`.

# Compile model as TensorRT from UI
Open `Model Catalog` from the dashboard, upload the model file if it is not already listed, then click the model's compile action to create a TensorRT engine. Wait for the compile job to finish, then return to the dashboard and select the compiled engine from the model dropdown.

# Development tips&tricks

Build the runtime image from the single Dockerfile:
```
cd ${REPO_DIR}
sudo docker build -t tilletia-app:latest -f docker/Dockerfile.desktop .
```

This image includes OpenCV, the pinned LINEA runtime, and the application code. Use `${REPO_DIR}` as the build context, not `${REPO_DIR}/docker`, because the image copies files from `src/`.

Optional: test the docker image by trying to compile as TensorRT the pre-trained Ultralytics YOLO model. You should see `TensorRT: export success ✅ 200.1s, saved as 'yolo11n.engine' (11.9 MB)` at the end:
```
docker run --network host --runtime=nvidia --rm -it -e NVIDIA_DRIVER_CAPABILITIES=all -v $(pwd):/app tilletia-app:latest yolo export format=engine
```

Ensure NVIDIA Container Toolkit is installed on the host if you run Docker with GPU acceleration.

# How to compile YOLO-model
Model must be compiled as tensorrt (once new model is added). Ultralytics models are stored in `data/model/ul` and Roboflow models in `data/model/rf`.

1. Download the model - open model card from the list and choose `Download Weights` button
![alt text](docs/download_model.png)
2. By default model is downloaded as `weights.pt`, rename it to the meaningful name, for example `yolo11-tilletia-detection-yolov8-seg-twxa6-41-fp16.pt` would be good to track the model type and origin from the app. The postfix `tilletia-detection-yolov8-seg-twxa6-41` is based on `Model URL` on the screenshot on previous step. `-fp16` is essential at the end of the filename to let the application find the model.
3. Copy the renamed model to the folder `${REPO_DIR}/data/model/ul/`

4. Compile the model to tensorrt
```
cd ${REPO_DIR}
docker run --network host --runtime=nvidia --rm -it -e NVIDIA_DRIVER_CAPABILITIES=all -v "$(pwd)/data/model:/app/model" tilletia-app:latest yolo export format=engine model=/app/model/ul/yolo11-tilletia-detection-yolov8-seg-twxa6-41-fp16.pt imgsz=640 half
```

The `/models` page can also manage model artifacts directly: compile missing engines, choose the Ultralytics task used at runtime (`segment`, `detect`, `auto`), and delete all artifacts for a model with confirmation.

To convert Roboflow RF-DETR object detection models:
```
cd ${REPO_DIR}
docker run --runtime nvidia --rm -it --entrypoint python3 -v "$(pwd)/src:/app" -v "$(pwd)/data/model:/app/model" export-rf /app/convert.py
```

# How to Run Application
Run mediamtx docker container (stop and remove if it's running)
```
cd ${REPO_DIR}
docker run --rm -d --name mediamtx --network host -v "$(pwd)/config/mediamtx.yml:/mediamtx.yml:ro" bluenviron/mediamtx:latest
```

Run app
```
cd ${REPO_DIR}
docker run --network host --runtime=nvidia --rm -it --device=/dev/video0 -v "$(pwd)/src:/app" -v "$(pwd)/data/model:/app/model" \
  -v "$(pwd)/data/output_hq:/app/output_hq" -v "$(pwd)/data/runs:/app/runs" tilletia-app:latest python /app/app.py
```


# How to Update the API readme file
Run application as described in previous section, then execute:
```
cd ${REPO_DIR}
curl http://localhost:8000/apispec_1.json > auto_swagger.json
python3 generate_docs.py
```

# Very small integration tests
Start the app first, then run:
```
cd ${REPO_DIR}
python3 -m unittest discover -s tests
```

The tests assume local auth exists, bearer access tokens are used, and the default admin user is `admin/admin`.
