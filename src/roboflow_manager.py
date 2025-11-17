from time import sleep
import uuid
from queue import Queue
import threading
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame


class RoboflowManager:
    def __init__(self, api_key: str, workspace: str, workflow_id: str) -> None:
        self.api_key = api_key
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.pipeline_id = None
        self.pipeline = None
        self.queue = None
        self.thread = None

    def list_pipelines(self):
        if self.pipeline_id:
            return [{'pipeline_id': self.pipeline_id}]
        return []

    def start_pipeline(self, input_rtsp: str):
        self.queue = Queue()

        def my_sink(predictions: dict, video_frame: VideoFrame):
            self.queue.put(predictions)

        self.pipeline = InferencePipeline.init_with_workflow(
            video_reference=input_rtsp,
            workspace_name=self.workspace,
            workflow_id=self.workflow_id,
            on_prediction=my_sink,
            api_key=self.api_key,
        )

        def run_pipeline():
            self.pipeline.start()
            self.pipeline.join()

        self.thread = threading.Thread(target=run_pipeline)
        self.thread.start()
        self.pipeline_id = str(uuid.uuid4())
        return self.pipeline_id

    def stop_pipeline(self, id=None) -> None:
        id = self.pipeline_id or id
        if not id:
            return
        
        if self.pipeline:
            self.pipeline.terminate()
            if self.thread:
                self.thread.join()
            self.pipeline = None
            self.queue = None
            self.thread = None
            self.pipeline_id = None

    def attach_to_pipeline(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        return self.pipeline_id

    def consume(self, pipeline_id: str = None, excluded_fields: list = None):
        pipeline_id = pipeline_id or self.pipeline_id
        if not pipeline_id:
            raise ValueError("Pipeline ID not set")
        if not self.queue:
            raise ValueError("Queue not initialized")
        return {'outputs': self.queue.get()}


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--api', type=str)
    args = args.parse_args()

    rf = RoboflowManager(args.api, "agroprosperis", "detect-count-and-visualize-c1-video-2")

    print(rf.list_pipelines())
    print(rf.start_pipeline('rtsp://127.0.0.1:8554/stream'))

    while True:
        sleep(1)
        results = rf.consume()
        print(list(results.get('outputs', {}).keys()))
        print(results.get('outputs', {'total_unique_objects_count': None})['total_unique_objects_count'])