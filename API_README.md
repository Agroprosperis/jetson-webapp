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
      "task": "segment",
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

## Delete all stored artifacts for a catalog model.
**POST** `/api/model-delete`



### Request Sample
```shell
curl -X POST "http://localhost/api/model-delete" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"name": "string", "type": "ul"}'
```

### Response
**200 OK**: Deleted artifact list
```json
{
  "deleted": [
    "string"
  ],
  "name": "string",
  "type": "string"
}
```

---

## Save the Ultralytics task override for a catalog model.
**POST** `/api/model-task`



### Request Sample
```shell
curl -X POST "http://localhost/api/model-task" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"name": "string", "task": "segment", "type": "ul"}'
```

### Response
**200 OK**: Saved task override
```json
{
  "name": "string",
  "task": "segment",
  "type": "string"
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



### Parameters
- `date` (query, string, optional) - Optional exact date in YYYY-MM-DD format.

### Request Sample
```shell
curl -X GET "http://localhost/api/results?date=string" \
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



### Parameters
- `analysis_id` (query, string, optional) - The Analysis ID or Timestamp to filter by
- `date` (query, string, optional) - Optional exact date in YYYY-MM-DD format.

### Request Sample
```shell
curl -X GET "http://localhost/api/results/search?analysis_id=string&date=string" \
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



### Parameters
- `pid` (path, string, required) - The Analysis ID to delete

### Request Sample
```shell
curl -X DELETE "http://localhost/api/results/string" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Files deleted successfully
---

## Download a result folder as a ZIP archive.
**GET** `/api/results/{pid}/download`



### Parameters
- `pid` (path, string, required) - The Analysis ID to download

### Request Sample
```shell
curl -X GET "http://localhost/api/results/string/download" \
  -H "accept: application/json" \
```

### Response
**200 OK**: ZIP archive of the result folder
---

## Return the last non-empty row from the result CSV as JSON.
**GET** `/api/results/{pid}/last-row`



### Parameters
- `pid` (path, string, required) - The Analysis ID to inspect

### Request Sample
```shell
curl -X GET "http://localhost/api/results/string/last-row" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Last CSV row
```json
{
  "analysis_id": "string",
  "row": {
    "analysis_number": "string",
    "detections": [
      {
        "bbox": [
          0
        ],
        "class_id": 0,
        "confidence": 0.0
      }
    ],
    "frame": 0,
    "s_value": 0.0,
    "total_unique_objects": 0
  }
}
```

---

## Start the inference pipeline.
**POST** `/api/start`



### Request Sample
```shell
curl -X POST "http://localhost/api/start" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"analysis_number": "string", "device": "string", "fps": 0, "grid_count_enabled": true, "grid_score_threshold": 0.0, "height": 0, "model_path": "string", "model_task": "segment", "source_type": "camera", "video": "string", "vis_conf": 0.0, "width": 0}'
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



### Parameters
- `file` (formData, file, required) - The video file to upload

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



### Parameters
- `filename` (path, string, required) - Relative file path within the results directory

### Request Sample
```shell
curl -X GET "http://localhost/download/string" \
  -H "accept: application/json" \
```

### Response
**200 OK**: File download
---
