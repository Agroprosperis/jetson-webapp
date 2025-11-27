# Setup step
REPO_DIR is the directory where the github repository is cloned
```
export REPO_DIR='/home/jetson/app'
```

On the very first run build docker image with all dependencies:
```
cd ${REPO_DIR}/docker
# for desktop tests
docker build -t opencv-gst:latest -f Dockerfile.desktop .

# for jetson production usage
docker build -t opencv-gst:latest -f Dockerfile.jetson-orin-nano .
```

Optional: test the docker image by trying to compile as tensorrt the pre-trained ultralytics yolon model. You should see `TensorRT: export success ✅ 200.1s, saved as 'yolo11n.engine' (11.9 MB)` at the end:
```
docker run --network host --runtime=nvidia --rm -it -e NVIDIA_DRIVER_CAPABILITIES=all -v $(pwd):/app opencv-gst:latest yolo export format=engine
```

Ensure host dependencies are installed:
```
sudo apt-get update
sudo apt-get install nvidia-l4t-dla-compiler
```

Model must be compiled as tensorrt (once new model is added). Ultralytics tensorrt models must be stored in `model/ul` folder and Roboflow models in `model/rf` folder
```
# Command must be executed on target device, first download pre-trained model and save as *.pt file
# in the following example pre-trained model is saved as cd ${REPO_DIR}/src/model/ul/yolo11s-seg-v15-fp16.pt
cd ${REPO_DIR}/src
docker run --network host --runtime=nvidia --rm -it -e NVIDIA_DRIVER_CAPABILITIES=all -v $(pwd):/app opencv-gst:latest yolo export format=engine model=/app/model/ul/yolo11s-seg-v15-fp16.pt imgsz=640 half
mv train17/weights/last.engine model/weights-fp16.engine
```

To convert Roboflow RF-DETR object detection models:
```
cd ${REPO_DIR}/docker
docker build -t export-rf -f Dockerfile.export-jetson .

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