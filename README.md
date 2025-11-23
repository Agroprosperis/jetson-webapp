# Setup step
REPO_DIR is the directory where the github repository is cloned

On the very first run build docker image with all dependencies:
```
cd ${REPO_DIR}/docker
# for desktop tests
docker build -t opencv-gst:latest -f Dockerfile.desktop .

# for jetson production usage
docker build -t opencv-gst:latest -f Dockerfile.jetson-orin-nano .
```

Model must be compiled as tensorrt and saved as src/model/weights-fp16.engine:
```
# Command must be executed on target device, first download pre-trained model and save as *.pt file
# in the followin example pre-trained model is saved as cd ${REPO_DIR}/src/runs/detect/train18/weights/last.pt
cd ${REPO_DIR}/src
docker run --network host --gpus all --rm -it -e NVIDIA_DRIVER_CAPABILITIES=all --device=/dev/video0 -v $(pwd):/app opencv-gst:latest yolo export format=engine model=/app/runs/detect/train18/weights/last.pt
mv /app/runs/detect/train18/weights/last.engine /app/src/model/weights-fp16.engine
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
docker run --network host --gpus all --rm -it --device=/dev/video0 -v $(pwd):/app opencv-gst:latest python /app/app.py
```