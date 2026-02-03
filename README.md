# Setup step
REPO_DIR is the directory where the github repository is cloned
```
export REPO_DIR='/home/${USER}/app'
```

On the very first run build docker image with all dependencies:
```
cd ${REPO_DIR}/docker
docker build -t opencv-gst:latest -f Dockerfile.desktop .
```

Optional: test the docker image by trying to compile as tensorrt the pre-trained ultralytics yolon model. You should see `TensorRT: export success ✅ 200.1s, saved as 'yolo11n.engine' (11.9 MB)` at the end:
```
docker run --network host --runtime=nvidia --rm -it -e NVIDIA_DRIVER_CAPABILITIES=all -v $(pwd):/app opencv-gst:latest yolo export format=engine
```

Ensure NVIDIA Container Toolkit is installed on the host if you run Docker with GPU acceleration.

# How to compile YOLO-model
Model must be compiled as tensorrt (once new model is added). Ultralytics tensorrt models must be stored in `model/ul` folder and Roboflow models in `model/rf` folder

1. Download the model - open model card from the list and choose `Download Weights` button
![alt text](docs/download_model.png)
2. By default model is downloaded as `weights.pt`, rename it to the meaningful name, for example `yolo11-tilletia-detection-yolov8-seg-twxa6-41-fp16.pt` would be good to track the model type and origin from the app. The postfix `tilletia-detection-yolov8-seg-twxa6-41` is based on `Model URL` on the screenshot on previous step. `-fp16` is essential at the end of the filename to let the application find the model.
3. Copy the renamed model to the folder ${REPO_DIR}/src/model/ul/

4. Compile the model to tensorrt
```
cd ${REPO_DIR}/src
docker run --network host --runtime=nvidia --rm -it -e NVIDIA_DRIVER_CAPABILITIES=all -v $(pwd):/app opencv-gst:latest yolo export format=engine model=/app/model/ul/yolo11-tilletia-detection-yolov8-seg-twxa6-41-fp16.pt imgsz=640 half
```

To convert Roboflow RF-DETR object detection models:
```
cd ${REPO_DIR}/src
docker run --runtime nvidia --rm -it --entrypoint python3 -v $(pwd):/app export-rf /app/convert.py
```

# How to Run Application
Run mediamtx docker container (stop and remove if it's running)
```
cd ${REPO_DIR}/src
docker run --rm -d --name mediamtx --network host -v "$(pwd)/mediamtx/mediamtx.yml:/mediamtx.yml:ro" bluenviron/mediamtx:latest
```

Run app
```
cd ${REPO_DIR}/src
docker run --network host --runtime=nvidia --rm -it --device=/dev/video0 -v $(pwd):/app opencv-gst:latest python /app/app.py
```

# How to Update the API readme file
Run application as described in previous section and execute
```
cd ${REPO_DIR}
curl http://localhost:8000/apispec_1.json > auto_swagger.json
python3 generate_docs.py
```
