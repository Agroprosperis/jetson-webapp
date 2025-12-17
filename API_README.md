---
title: Jetson Inference Pipeline API v0.0.1
language_tabs:
  - shell: Curl
language_clients:
  - shell: ""
toc_footers: []
includes: []
search: false
highlight_theme: darkula
headingLevel: 2

---

<!-- Generator: Widdershins v4.0.1 -->

<h1 id="jetson-inference-pipeline-api">Jetson Inference Pipeline API v0.0.1</h1>

> Scroll down for code samples, example requests and responses. Select a language for code samples from the tabs above or the mobile navigation menu.

powered by Flasgger

<h1 id="jetson-inference-pipeline-api-configuration">Configuration</h1>

## List attached cameras and their modes.

> Code samples

```shell
# You can also use wget
curl -X GET /api/cameras \
  -H 'Accept: */*'

```

`GET /api/cameras`

> Example responses

> 200 Response

<h3 id="list-attached-cameras-and-their-modes.-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|List of V4L2 devices and supported modes|Inline|

<h3 id="list-attached-cameras-and-their-modes.-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» cameras|[object]|false|none|none|
|»» device|string|false|none|none|
|»» modes|[object]|false|none|none|
|»» name|string|false|none|none|

<aside class="success">
This operation does not require authentication
</aside>

## List available *.engine models.

> Code samples

```shell
# You can also use wget
curl -X GET /api/models \
  -H 'Accept: */*'

```

`GET /api/models`

> Example responses

> 200 Response

<h3 id="list-available-*.engine-models.-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|List of available TensorRT engine files|Inline|

<h3 id="list-available-*.engine-models.-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» models|[object]|false|none|none|
|»» name|string|false|none|none|
|»» path|string|false|none|none|
|»» type|string|false|none|none|

<aside class="success">
This operation does not require authentication
</aside>

## Upload a video file.

> Code samples

```shell
# You can also use wget
curl -X POST /api/upload \
  -H 'Content-Type: multipart/form-data' \
  -H 'Accept: */*'

```

`POST /api/upload`

> Body parameter

```yaml
file: string

```

<h3 id="upload-a-video-file.-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|body|body|object|true|none|
|» file|body|string(binary)|true|The video file to upload|

> Example responses

> 200 Response

<h3 id="upload-a-video-file.-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|File uploaded successfully|Inline|

<h3 id="upload-a-video-file.-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» video|string|false|none|Server path to the uploaded file|

<aside class="success">
This operation does not require authentication
</aside>

<h1 id="jetson-inference-pipeline-api-results">Results</h1>

## List all results.

> Code samples

```shell
# You can also use wget
curl -X GET /api/results \
  -H 'Accept: */*'

```

`GET /api/results`

> Example responses

> 200 Response

<h3 id="list-all-results.-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|List of processed videos|Inline|

<h3 id="list-all-results.-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» results|[object]|false|none|none|
|»» id|string|false|none|none|
|»» timestamp|string|false|none|none|
|»» video_size|string|false|none|none|

<aside class="success">
This operation does not require authentication
</aside>

## Search results by Analysis ID.

> Code samples

```shell
# You can also use wget
curl -X GET /api/results/search \
  -H 'Accept: */*'

```

`GET /api/results/search`

<h3 id="search-results-by-analysis-id.-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|analysis_id|query|string|false|The Analysis ID or Timestamp to filter by|

> Example responses

> 200 Response

<h3 id="search-results-by-analysis-id.-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|List of matching results with download links|Inline|

<h3 id="search-results-by-analysis-id.-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» results|[object]|false|none|none|
|»» analysis_id|string|false|none|none|
|»» csv_url|string|false|none|none|
|»» video_url|string|false|none|none|

<aside class="success">
This operation does not require authentication
</aside>

## Delete a result set.

> Code samples

```shell
# You can also use wget
curl -X DELETE /api/results/{pid}

```

`DELETE /api/results/{pid}`

<h3 id="delete-a-result-set.-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|pid|path|string|true|The Analysis ID to delete|

<h3 id="delete-a-result-set.-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Files deleted successfully|None|

<aside class="success">
This operation does not require authentication
</aside>

<h1 id="jetson-inference-pipeline-api-control">Control</h1>

## Start the inference pipeline.

> Code samples

```shell
# You can also use wget
curl -X POST /api/start \
  -H 'Content-Type: application/json' \
  -H 'Accept: */*'

```

`POST /api/start`

> Body parameter

```json
{
  "analysis_number": "string",
  "device": "string",
  "fps": 0,
  "height": 0,
  "model_path": "string",
  "source_type": "camera",
  "video": "string",
  "vis_conf": 0,
  "width": 0
}
```

<h3 id="start-the-inference-pipeline.-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|body|body|object|true|none|
|» analysis_number|body|string|false|Optional custom ID|
|» device|body|string|false|Device path (e.g., /dev/video0)|
|» fps|body|integer|false|none|
|» height|body|integer|false|none|
|» model_path|body|string|false|none|
|» source_type|body|string|false|none|
|» video|body|string|false|Path to uploaded video file|
|» vis_conf|body|number|false|none|
|» width|body|integer|false|none|

#### Enumerated Values

|Parameter|Value|
|---|---|
|» source_type|camera|
|» source_type|file|

> Example responses

> 200 Response

<h3 id="start-the-inference-pipeline.-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Pipeline started successfully|Inline|
|400|[Bad Request](https://tools.ietf.org/html/rfc7231#section-6.5.1)|Already running or invalid input|None|

<h3 id="start-the-inference-pipeline.-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» pipeline_id|string|false|none|none|
|» success|boolean|false|none|none|

<aside class="success">
This operation does not require authentication
</aside>

## Get pipeline status.

> Code samples

```shell
# You can also use wget
curl -X GET /api/status \
  -H 'Accept: */*'

```

`GET /api/status`

> Example responses

> 200 Response

<h3 id="get-pipeline-status.-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Current state, threads, and MediaMTX status|Inline|

<h3 id="get-pipeline-status.-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» mediamtx|object|false|none|none|
|»» rtsp|boolean|false|none|none|
|»» whep|boolean|false|none|none|
|» pipeline_id|string|false|none|none|
|» state|string|false|none|none|

#### Enumerated Values

|Property|Value|
|---|---|
|state|idle|
|state|running|

<aside class="success">
This operation does not require authentication
</aside>

## Stop the inference pipeline.

> Code samples

```shell
# You can also use wget
curl -X POST /api/stop

```

`POST /api/stop`

<h3 id="stop-the-inference-pipeline.-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Pipeline stopped|None|

<aside class="success">
This operation does not require authentication
</aside>

