# Jetson Inference Pipeline API

**Version:** 0.0.1

**Description:** powered by Flasgger

**Terms of Service:** /tos

---

## List attached cameras and their modes.
**GET** `/api/cameras`



### Request Sample
```shell
curl -X GET "http://localhost/api/cameras" \
  -H "accept: application/json" \
```

### Response
**200 OK**: List of V4L2 devices and supported modes
```json
{
  "cameras": [
    {
      "device": "string",
      "modes": [
        {}
      ],
      "name": "string"
    }
  ]
}
```

---

## Get basic runtime configuration.
**GET** `/api/config`



### Request Sample
```shell
curl -X GET "http://localhost/api/config" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Basic configuration values
```json
{
  "stream_port": 0
}
```

---

## List available *.engine models.
**GET** `/api/models`



### Request Sample
```shell
curl -X GET "http://localhost/api/models" \
  -H "accept: application/json" \
```

### Response
**200 OK**: List of available TensorRT engine files
```json
{
  "models": [
    {
      "name": "string",
      "path": "string",
      "type": "string"
    }
  ]
}
```

---

## List all results.
**GET** `/api/results`



### Request Sample
```shell
curl -X GET "http://localhost/api/results" \
  -H "accept: application/json" \
```

### Response
**200 OK**: List of processed videos
```json
{
  "results": [
    {
      "id": "string",
      "timestamp": "string",
      "video_size": "string"
    }
  ]
}
```

---

## Search results by Analysis ID.
**GET** `/api/results/search`



### Request Sample
```shell
curl -X GET "http://localhost/api/results/search" \
  -H "accept: application/json" \
```

### Response
**200 OK**: List of matching results with download links
```json
{
  "results": [
    {
      "analysis_id": "string",
      "csv_url": "string",
      "video_url": "string"
    }
  ]
}
```

---

## Delete a result set.
**DELETE** `/api/results/{pid}`



### Request Sample
```shell
curl -X DELETE "http://localhost/api/results/{pid}" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Files deleted successfully
---

## Download a result folder as a ZIP archive.
**GET** `/api/results/{pid}/download`



### Request Sample
```shell
curl -X GET "http://localhost/api/results/{pid}/download" \
  -H "accept: application/json" \
```

### Response
**200 OK**: ZIP archive of the result folder
---

## Start the inference pipeline.
**POST** `/api/start`



### Request Sample
```shell
curl -X POST "http://localhost/api/start" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"analysis_number": "string", "device": "string", "fps": 0, "height": 0, "model_path": "string", "source_type": "camera", "video": "string", "vis_conf": 0.0, "width": 0}'
```

### Response
**200 OK**: Pipeline started successfully
```json
{
  "pipeline_id": "string",
  "success": true
}
```

---

## Get pipeline status.
**GET** `/api/status`



### Request Sample
```shell
curl -X GET "http://localhost/api/status" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Current state, threads, and MediaMTX status
```json
{
  "mediamtx": {
    "rtsp": true,
    "whep": true
  },
  "pipeline_id": "string",
  "state": "idle"
}
```

---

## Stop the inference pipeline.
**POST** `/api/stop`



### Request Sample
```shell
curl -X POST "http://localhost/api/stop" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Pipeline stopped
---

## Upload a video file.
**POST** `/api/upload`



### Request Sample
```shell
curl -X POST "http://localhost/api/upload" \
  -H "accept: application/json" \
```

### Response
**200 OK**: File uploaded successfully
```json
{
  "video": "string"
}
```

---

## Download a file from the HQ output directory.
**GET** `/download/{filename}`



### Request Sample
```shell
curl -X GET "http://localhost/download/{filename}" \
  -H "accept: application/json" \
```

### Response
**200 OK**: File download
---
