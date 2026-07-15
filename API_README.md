# Desktop Inference Pipeline API

**Version:** 0.0.1

**Description:** powered by Flasgger

**Terms of Service:** /tos

## Authentication

- Log in with `POST /auth/login` to obtain an `access_token` and `refresh_token`.
- Send the access token on protected endpoints with `Authorization: Bearer <access_token>`.
- Refresh expired access tokens with `POST /auth/refresh`.

---

## Serve the main dashboard page.
**GET** `/`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Dashboard HTML.
---

## List attached cameras and their modes.
**GET** `/api/cameras`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/cameras" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: List of V4L2 devices and supported modes
```json
{
  "cameras": [
    {
      "device": "string",
      "modes": [
        {
          "format": "string",
          "fps": 0,
          "height": 0,
          "width": 0
        }
      ],
      "name": "string"
    }
  ]
}
```

---

## Get basic runtime configuration.
**GET** `/api/config`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/config" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Basic configuration values
```json
{
  "stream_port": 0
}
```

---

## Get the persisted dashboard settings used as defaults for new runs.
**GET** `/api/dashboard-settings`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/dashboard-settings" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Current dashboard settings.
```json
{
  "analysis_number": "string",
  "ask_manual_spore_count": true,
  "camera_device": "string",
  "camera_mode": {
    "format": "string",
    "fps": 0,
    "height": 0,
    "width": 0
  },
  "captions_enabled": true,
  "grid_count_enabled": true,
  "grid_debug_enabled": true,
  "grid_score_threshold": 0.0,
  "model_path": "string",
  "source_type": "camera",
  "uploaded_path": "string",
  "vis_conf": 0.0
}
```

---

## Update the persisted dashboard settings used as defaults for new runs.
**PUT** `/api/dashboard-settings`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X PUT "http://localhost:8000/api/dashboard-settings" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"analysis_number": "string", "ask_manual_spore_count": true, "camera_device": "string", "camera_mode": {"format": "string", "fps": 0, "height": 0, "width": 0}, "captions_enabled": true, "grid_count_enabled": true, "grid_debug_enabled": true, "grid_score_threshold": 0.0, "model_path": "string", "source_type": "camera", "uploaded_path": "string", "vis_conf": 0.0}'
```

### Response
**200 OK**: Updated dashboard settings.
---

## Get the current grid feature state.
**GET** `/api/grid`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/grid" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Current grid feature state
```json
{
  "auto_disabled": true,
  "debug_enabled": true,
  "enabled": true,
  "score": 0.0,
  "score_threshold": 0.0
}
```

---

## Update grid feature settings.
**PUT** `/api/grid`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X PUT "http://localhost:8000/api/grid" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"debug_enabled": true, "enabled": true, "score_threshold": 0.0}'
```

### Response
**200 OK**: Updated grid feature state
```json
{
  "auto_disabled": true,
  "debug_enabled": true,
  "enabled": true,
  "score": 0.0,
  "score_threshold": 0.0
}
```

---

## List available model sources and their TensorRT engines (if compiled).
**GET** `/api/model-catalog`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/model-catalog" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: List of model files and their TensorRT engines
```json
{
  "models": [
    {
      "compiled": true,
      "default_confidence_threshold": 0.0,
      "engine": {
        "name": "string",
        "path": "string"
      },
      "name": "string",
      "owner_username": "string",
      "source_paths": [
        "string"
      ],
      "sources": [
        "string"
      ],
      "task": "segment",
      "tilletia_filter_max_height_px": 0,
      "tilletia_filter_max_width_px": 0,
      "tilletia_filter_training_height": 0,
      "tilletia_filter_training_width": 0,
      "type": "string"
    }
  ],
  "tensorrt": {
    "current": "string"
  }
}
```

---

## Start async model compilation to TensorRT (FP16).
**POST** `/api/model-compile`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/api/model-compile" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"inference_height": 0, "inference_width": 0, "name": "string", "type": "ul"}'
```

### Response
**200 OK**: Compile job queued
```json
{
  "already_running": true,
  "job_id": "string"
}
```

---

## List compile jobs for UI restore after close/refresh.
**GET** `/api/model-compile-jobs`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/model-compile-jobs" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Compile job list
```json
{
  "jobs": [
    {
      "created_at": "string",
      "finished_at": "string",
      "id": "string",
      "model": {
        "name": "string",
        "source": "string",
        "type": "string"
      },
      "returncode": 0,
      "started_at": "string",
      "status": "string"
    }
  ]
}
```

---

## Get compile job status and logs.
**GET** `/api/model-compile/{job_id}`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `job_id` (path, string, required) - Compile job identifier returned by `/api/model-compile`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/model-compile/string" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Compile job status
```json
{
  "command": [
    "string"
  ],
  "created_at": "string",
  "finished_at": "string",
  "id": "string",
  "logs": [
    "string"
  ],
  "model": {},
  "returncode": 0,
  "started_at": "string",
  "status": "string"
}
```

---

## Delete all stored artifacts for a catalog model.
**POST** `/api/model-delete`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/api/model-delete" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
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

## Save editable metadata for a catalog model.
**POST** `/api/model-metadata`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/api/model-metadata" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"default_confidence_threshold": 0.0, "name": "string", "tilletia_filter_max_height_px": 0, "tilletia_filter_max_width_px": 0, "tilletia_filter_training_height": 0, "tilletia_filter_training_width": 0, "type": "ul"}'
```

### Response
**200 OK**: Saved model metadata
---

## Save the Ultralytics task override for a catalog model.
**POST** `/api/model-task`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/api/model-task" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
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

## Upload a model weights file or RF deployment package into the catalog.
**POST** `/api/model-upload`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `type` (formData, string, required): allowed values `ul, rf` - Target model family for the uploaded weights file.
- `file` (formData, file, required) - Ultralytics/RF-DETR `.pt` weights, or an RF deployment package `.zip`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/api/model-upload" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Model uploaded successfully
```json
{
  "tensorrt": {
    "current": "string"
  },
  "uploaded": {
    "name": "string",
    "path": "string",
    "type": "string"
  }
}
```

---

## List available *.engine models.
**GET** `/api/models`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/models" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: List of available TensorRT engine files
```json
{
  "models": [
    {
      "default_confidence_threshold": 0.0,
      "display": "string",
      "name": "string",
      "owner_username": "string",
      "path": "string",
      "type": "string"
    }
  ]
}
```

---

## List all results using the deprecated March 22 v1 shape.
**GET** `/api/results`

Deprecated compatibility endpoint matching the March 22 API. The result list only exposes id, timestamp, and video_size. Use /api/v2/results for duration, files, owner_username, and per-class metrics.


**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `date` (query, string, optional) - Optional exact date in YYYY-MM-DD format.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/results?date=string" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
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



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `analysis_id` (query, string, optional) - The Analysis ID or Timestamp to filter by
- `date` (query, string, optional) - Optional exact date in YYYY-MM-DD format.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/results/search?analysis_id=string&date=string" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: List of matching results with download links
```json
{
  "results": [
    {
      "analysis_id": "string",
      "csv_url": "string",
      "images": [
        {
          "name": "string",
          "size": "string",
          "url": "string"
        }
      ],
      "video_url": "string"
    }
  ]
}
```

---

## Delete a result set.
**DELETE** `/api/results/{pid}`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `pid` (path, string, required) - The Analysis ID to delete

### Request Sample
```shell
curl -X DELETE "http://localhost:8000/api/results/string" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Files deleted successfully
```json
{
  "deleted": [
    "string"
  ],
  "success": true
}
```

---

## Download a result folder as a ZIP archive.
**GET** `/api/results/{pid}/download`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `pid` (path, string, required) - The Analysis ID to download

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/results/string/download" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: ZIP archive of the result folder
---

## Return the last non-empty row from the result CSV using the deprecated v1 metric shape.
**GET** `/api/results/{pid}/last-row`

Deprecated compatibility endpoint. The CSV row is normalized into the old scalar fields: total_unique_objects and s_value. Current per-class CSV rows are collapsed by summing detected_objects_per_class and calculating the legacy all-object S value. Use /api/v2/results/{pid}/last-row for per-class metrics.


**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `pid` (path, string, required) - The Analysis ID to inspect

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/results/string/last-row" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
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

## Store the manually counted actual smut spore numbers for a result.
**PUT** `/api/results/{pid}/manual-count`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `pid` (path, string, required) - The Analysis ID to update

### Request Sample
```shell
curl -X PUT "http://localhost:8000/api/results/string/manual-count" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Manual counts stored successfully
```json
{
  "success": true
}
```

---

## Reuse a processed result video as the current dashboard file source.
**POST** `/api/results/{pid}/process-source`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/api/results/{pid}/process-source" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Result video prepared as the dashboard file source.
```json
{
  "file_name": "string",
  "success": true,
  "video": "string"
}
```

---

## Capture a manual snapshot from the active pipeline.
**POST** `/api/snapshot`

<br/>Optional JSON body for zoomed snapshots:<br/>- `zoom_level` controls whether the snapshot is saved as a zoom crop. Use a<br/>  finite number greater than 1 to apply the crop. Values less than or equal<br/>  to 1 save the full current frame. There is no API-enforced maximum.<br/>- `crop.x` and `crop.y` are the left and top edges of the visible region in<br/>  source-frame pixels. Negative or out-of-frame values are accepted and<br/>  clamped to the frame.<br/>- `crop.width` and `crop.height` are the crop size in source-frame pixels.<br/>  They must be finite numbers greater than 0 to apply a zoom crop; otherwise<br/>  the full current frame is saved. Oversized values are clamped to the frame.<br/>- Omit the body, send `{}`, or omit either `zoom_level` or `crop` to save the<br/>  full current frame. If `crop` is present with `zoom_level`, all four crop<br/>  fields are required. Fractional crop values are accepted and rounded when<br/>  the JPEG is written.<br/>

**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/api/snapshot" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"crop": {"height": 0.0, "width": 0.0, "x": 0.0, "y": 0.0}, "zoom_level": 0.0}'
```

### Response
**200 OK**: Snapshot saved successfully.
```json
{
  "filename": "string",
  "path": "string",
  "pipeline_id": "string",
  "success": true
}
```

---

## Start the inference pipeline.
**POST** `/api/start`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/api/start" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"analysis_number": "string", "captions_enabled": true, "device": "string", "format": "string", "fps": 0, "grid_count_enabled": true, "grid_debug_enabled": true, "grid_score_threshold": 0.0, "height": 0, "model_path": "string", "model_task": "segment", "source_type": "camera", "video": "string", "vis_conf": 0.0, "vis_strategy": "string", "width": 0}'
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



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/status" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Current state, threads, and MediaMTX status
```json
{
  "config": {
    "model": "string",
    "model_task": "string",
    "video_reference": "string"
  },
  "last_error": "string",
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
  "state": "idle",
  "threads": [
    "string"
  ]
}
```

---

## Stop the inference pipeline.
**POST** `/api/stop`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/api/stop" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Pipeline stopped
```json
{
  "success": true
}
```

---

## Upload a video file.
**POST** `/api/upload`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `file` (formData, file, required) - The video file to upload

### Request Sample
```shell
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: File uploaded successfully
```json
{
  "video": "string"
}
```

---

## List all results using the v2 per-class metric shape.
**GET** `/api/v2/results`

Result metrics are returned as per-class maps: detected_objects_per_class contains detected object counts and s_value_per_class contains S values, both keyed by the resolved model class name. Old-style single-class CSV rows with total_unique_objects and scalar s_value are Tilletia-only and are returned under the Tilletia key.


**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `date` (query, string, optional) - Optional exact date in YYYY-MM-DD format.
- `analysis_number` (query, string, optional) - Optional include filter for the analysis number/result ID.

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/v2/results?date=string&analysis_number=string" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: List of processed videos
```json
{
  "results": [
    {
      "detected_objects_per_class": {},
      "duration": "string",
      "duration_seconds": 0.0,
      "files": [
        {
          "name": "string",
          "path": "string",
          "size": "string"
        }
      ],
      "id": "string",
      "manual_spore_count_per_class": {},
      "owner_username": "string",
      "s_value_per_class": {},
      "timestamp": "string"
    }
  ]
}
```

---

## Return the last non-empty row from the result CSV using the v2 per-class metric shape.
**GET** `/api/v2/results/{pid}/last-row`

The CSV row is normalized before it is returned. Current CSV rows expose detected_objects_per_class and s_value_per_class as per-class maps keyed by the resolved model class name. Old-style single-class CSV rows with total_unique_objects and scalar s_value are Tilletia-only and are returned under the Tilletia key.


**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `pid` (path, string, required) - The Analysis ID to inspect

### Request Sample
```shell
curl -X GET "http://localhost:8000/api/v2/results/string/last-row" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Last CSV row
```json
{
  "analysis_id": "string",
  "row": {
    "analysis_number": "string",
    "detected_objects_per_class": {},
    "detections": [
      {
        "bbox": [
          0
        ],
        "class_id": 0,
        "class_name": "string",
        "confidence": 0.0
      }
    ],
    "frame": 0,
    "s_value_per_class": {}
  }
}
```

---

## Change a local user's password and clear the first-login password-change requirement.
**POST** `/auth/change-password`



**Auth:** No bearer token required. Provide `username`, `current_password`, `new_password`, and `confirm_password`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/auth/change-password" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"confirm_password": "string", "current_password": "string", "new_password": "string", "username": "string"}'
```

### Response
**200 OK**: Password updated successfully.
```json
{
  "success": true
}
```

---

## Authenticate a local user and issue bearer tokens.
**POST** `/auth/login`



**Auth:** No token required. Returns `access_token` and `refresh_token`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/auth/login" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"password": "string", "username": "string"}'
```

### Response
**200 OK**: Access token, refresh token, and authenticated user identity.
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "user": {
    "id": 0,
    "username": "string"
  }
}
```

---

## Revoke the current refresh token and clear auth cookies.
**POST** `/auth/logout`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/auth/logout" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Logout completed successfully.
```json
{
  "success": true
}
```

---

## Return the authenticated user identity.
**GET** `/auth/me`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/auth/me" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Authenticated user identity.
```json
{
  "id": 0,
  "username": "string"
}
```

---

## Exchange a refresh token for a new bearer access token.
**POST** `/auth/refresh`



**Auth:** No access token required. Provide the refresh token in the request body or refresh cookie.

### Request Sample
```shell
curl -X POST "http://localhost:8000/auth/refresh" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "string"}'
```

### Response
**200 OK**: Refreshed access token, rotated refresh token, and authenticated user identity.
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "user": {
    "id": 0,
    "username": "string"
  }
}
```

---

## Download a file from the HQ output directory.
**GET** `/download/{filename}`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `filename` (path, string, required) - Relative file path within the results directory

### Request Sample
```shell
curl -X GET "http://localhost:8000/download/string" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: File download
---

## Serve the login page.
**GET** `/login`



**Auth:** No access token required.

### Request Sample
```shell
curl -X GET "http://localhost:8000/login" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Login page HTML.
---

## Serve the model catalog page.
**GET** `/models`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/models" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Model catalog HTML.
---

## Serve the results page.
**GET** `/results`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/results" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Results page HTML.
---

## Serve the authenticated user settings page.
**GET** `/settings`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/settings" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: User settings HTML.
---

## Serve the user-management page.
**GET** `/users`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X GET "http://localhost:8000/users" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: User-management HTML.
---

## Create a local user with one or more roles.
**POST** `/users`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Request Sample
```shell
curl -X POST "http://localhost:8000/users" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"password": "string", "roles": ["admin"], "username": "string"}'
```

### Response
**200 OK**: Created user summary.
```json
{
  "id": 0,
  "username": "string"
}
```

---

## Serve bundled static frontend assets.
**GET** `/vendor/{filename}`



**Auth:** No access token required.

### Parameters
- `filename` (path, string, required) - Relative asset path within the bundled vendor directory.

### Request Sample
```shell
curl -X GET "http://localhost:8000/vendor/string" \
  -H "accept: application/json" \
```

### Response
**200 OK**: Static asset response.
---

## Proxy WHEP signaling requests to the local MediaMTX instance.
**GET** `/{path}/whep`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `path` (path, string, required) - Stream path forwarded to the upstream `/<path>/whep` endpoint.

### Request Sample
```shell
curl -X GET "http://localhost:8000/string/whep" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Successful proxied WHEP response.
---

## Proxy WHEP signaling requests to the local MediaMTX instance.
**POST** `/{path}/whep`



**Auth:** Bearer access token required. Add `-H "Authorization: Bearer <access_token>"`.

### Parameters
- `path` (path, string, required) - Stream path forwarded to the upstream `/<path>/whep` endpoint.

### Request Sample
```shell
curl -X POST "http://localhost:8000/string/whep" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <access_token>" \
```

### Response
**200 OK**: Successful proxied WHEP response.
---
