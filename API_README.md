# Desktop Inference Pipeline API

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

## Get the current grid feature state.
**GET** `/api/grid`



### Request Sample
```shell
curl -X GET "http://localhost/api/grid" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Current grid feature state
```json
{
  "auto_disabled": true,
  "enabled": true,
  "score": 0.0,
  "score_threshold": 0.0
}
```

---

## Update grid feature settings.
**PUT** `/api/grid`



### Request Sample
```shell
curl -X PUT "http://localhost/api/grid" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "score_threshold": 0.0}'
```

### Response
**200 OK**: Updated grid feature state
```json
{
  "auto_disabled": true,
  "enabled": true,
  "score": 0.0,
  "score_threshold": 0.0
}
```

---

## List available model sources and their TensorRT engines (if compiled).
**GET** `/api/model-catalog`



### Request Sample
```shell
curl -X GET "http://localhost/api/model-catalog" \
  -H "accept: application/json" \
```

### Response
**200 OK**: List of model files and their TensorRT engines
```json
{
  "models": [
    {
      "compiled": true,
      "engine": {
        "name": "string",
        "path": "string"
      },
      "name": "string",
      "source_paths": [
        "string"
      ],
      "sources": [
        "string"
      ],
      "type": "string"
    }
  ]
}
```

---

## Start async model compilation to TensorRT (FP16).
**POST** `/api/model-compile`



### Request Sample
```shell
curl -X POST "http://localhost/api/model-compile" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Compile job queued
---

## List compile jobs for UI restore after close/refresh.
**GET** `/api/model-compile-jobs`



### Request Sample
```shell
curl -X GET "http://localhost/api/model-compile-jobs" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Compile job list
---

## Get compile job status and logs.
**GET** `/api/model-compile/{job_id}`



### Request Sample
```shell
curl -X GET "http://localhost/api/model-compile/{job_id}" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Compile job status
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
  -d '{"analysis_number": "string", "device": "string", "fps": 0, "grid_count_enabled": true, "grid_score_threshold": 0.0, "height": 0, "model_path": "string", "source_type": "camera", "video": "string", "vis_conf": 0.0, "width": 0}'
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
  "runtime": {
    "grid_auto_disabled": true,
    "grid_count_enabled": true,
    "grid_score": 0.0,
    "grid_score_threshold": 0.0
  },
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
