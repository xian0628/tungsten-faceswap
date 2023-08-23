import os
import shutil
import warnings
from typing import List

import torch  # torch should be imported before insightface to use GPUs
from tungstenkit import BaseIO, Field, Image, Option, Video, define_model

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# single thread doubles cuda performance - needs to be set before torch import
os.environ["OMP_NUM_THREADS"] = "1"
# reduce tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import roop.globals
import roop.metadata
from roop.core import limit_resources, pre_check, suggest_execution_threads
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import (
    clean_temp,
    create_temp,
    create_video,
    detect_fps,
    extract_frames,
    get_temp_frame_paths,
    move_temp,
    normalize_output_path,
    restore_audio,
    run_ffmpeg,
)

# Put your video segments here
VIDEO_SEGMENTS = {
    "short": ["new-world-conversation.mp4"],
    "middle": ["new-world-conversation.mp4", "new-world-die.mp4"],
    "long": [
        "new-world-conversation.mp4",
        "new-world-die.mp4",
        "new-world-ev.mp4",
    ],
}
# Put indices of frames with face to swap
REFERENCE_FRAME_NUMBERS = {
    "short": [0],
    "middle": [0, 0],
    "long": [0, 0, 30],
}

OUTPUT_FILE_NAME = "output.mp4"


class FaceswapInput(BaseIO):
    face_image: Image = Field(description="Input face image")
    # You can customize output video types
    output_video_type: str = Field(
        description="Output video length", choices=["short", "middle", "long"]
    )


class FaceswapOutput(BaseIO):
    output_video: Video


@define_model(
    input=FaceswapInput,
    output=FaceswapOutput,
    system_packages=["python3-opencv", "ffmpeg"],
    python_packages=[
        "torch",
        "torchvision",
        "numpy==1.24.3",
        "opencv-python==4.8.0.74",
        "onnx==1.14.0",
        "insightface==0.7.3",
        "psutil==5.9.5",
        "pillow==10.0.0",
        "onnxruntime-gpu==1.15.1",
        "tensorflow==2.13.0",
        "opennsfw2==0.10.2",
        "protobuf==4.23.4",
        "tqdm==4.65.0",
        "gfpgan==1.3.8",
    ],
    gpu=True,
    cuda_version="11.8",
    python_version="3.9",
    force_install_system_cuda=True,
)
class FaceswapRoop:
    def setup(self):
        roop.globals.headless = True
        roop.globals.keep_fps = False
        roop.globals.keep_frames = False
        roop.globals.skip_audio = False
        roop.globals.many_faces = True
        roop.globals.reference_face_position = 0
        roop.globals.similar_face_distance = 0.85
        roop.globals.temp_frame_format = "png"
        roop.globals.temp_frame_quality = 0
        roop.globals.output_video_encoder = "libx264"
        roop.globals.output_video_quality = 35
        roop.globals.max_memory = None
        roop.globals.execution_providers = ["CUDAExecutionProvider"]
        roop.globals.execution_threads = suggest_execution_threads()
        if not pre_check():
            return
        for frame_processor in get_frame_processors_modules(
            ["face_swapper", "face_enhancer"]
        ):
            if not frame_processor.pre_start():
                return
        limit_resources()

    def predict(self, inputs: List[FaceswapInput]) -> List[FaceswapOutput]:
        input = inputs[0]

        roop.globals.total_videos = len(VIDEO_SEGMENTS[input.output_video_type])
        roop.globals.source_path = str(input.face_image.path)

        self._cleanup(input)

        for video_idx, video in enumerate(VIDEO_SEGMENTS[input.output_video_type]):
            roop.globals.video_idx = video_idx
            self._swap_face(input, video, video_idx)

        self._merge_video_segments(input)
        self._cleanup(input)
        return [FaceswapOutput(output_video=Video.from_path(OUTPUT_FILE_NAME))]

    def _swap_face(self, input: FaceswapInput, target_video_path: str, video_idx: int):
        roop.globals.frame_processors = ["face_swapper"]
        roop.globals.target_path = VIDEO_SEGMENTS[input.output_video_type][video_idx]
        roop.globals.output_path = normalize_output_path(
            roop.globals.source_path,
            target_video_path,
            f"output-segment-{video_idx}.mp4",
        )
        roop.globals.reference_frame_number = REFERENCE_FRAME_NUMBERS[
            input.output_video_type
        ][video_idx]

        # extract frames
        print(
            f"[Segment {video_idx+1}/{roop.globals.total_videos}] Extracting frames..."
        )
        create_temp(roop.globals.target_path)
        extract_frames(roop.globals.target_path)

        # process frame
        print(
            f"[Segment {video_idx+1}/{roop.globals.total_videos}] Processing frames..."
        )
        temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
        if temp_frame_paths:
            for frame_processor in get_frame_processors_modules(
                roop.globals.frame_processors
            ):
                frame_processor.process_video(
                    roop.globals.source_path, temp_frame_paths
                )
                frame_processor.post_process()
        else:
            print("Frames not found...")
            return

        # create video
        print(f"[Segment {video_idx+1}/{roop.globals.total_videos}] Creating video...")
        if roop.globals.keep_fps:
            fps = detect_fps(roop.globals.target_path)
            create_video(roop.globals.target_path, fps)
        else:
            create_video(roop.globals.target_path)

        # handle audio
        print(f"[Segment {video_idx+1}/{roop.globals.total_videos}] Adding audio...")
        if roop.globals.skip_audio:
            move_temp(roop.globals.target_path, roop.globals.output_path)
        else:
            restore_audio(roop.globals.target_path, roop.globals.output_path)

        # clean temp
        clean_temp(roop.globals.target_path)

    def _merge_video_segments(self, input: FaceswapInput):
        if len(VIDEO_SEGMENTS[input.output_video_type]) == 1:
            shutil.move("output-segment-0.mp4", OUTPUT_FILE_NAME)
        else:
            print("Merging videos...")
            if os.path.exists(OUTPUT_FILE_NAME):
                os.remove(OUTPUT_FILE_NAME)

            ffmpeg_args: List[str] = []
            for i in range(len(VIDEO_SEGMENTS[input.output_video_type])):
                ffmpeg_args.append("-i")
                ffmpeg_args.append(f"output-segment-{i}.mp4")

            ffmpeg_args.append("-filter_complex")
            ffmpeg_filter_args: List[str] = []
            for i in range(len(VIDEO_SEGMENTS[input.output_video_type])):
                ffmpeg_filter_args.append(f"[{i}:v]")
                ffmpeg_filter_args.append(f"[{i}:a]")
            ffmpeg_filter_args.append(
                f"concat=n={len(VIDEO_SEGMENTS[input.output_video_type])}:v=1:a=1 [v] [a]"
            )
            ffmpeg_args.append(" ".join(ffmpeg_filter_args))
            ffmpeg_args.append("-map")
            ffmpeg_args.append("[v]")
            ffmpeg_args.append("-map")
            ffmpeg_args.append("[a]")
            ffmpeg_args.append(OUTPUT_FILE_NAME)
            run_ffmpeg(ffmpeg_args)

    def _cleanup(self, input: FaceswapInput):
        for i in range(len(VIDEO_SEGMENTS[input.output_video_type])):
            segment_file_name = f"output-segment-{i}.mp4"
            if os.path.exists(segment_file_name):
                os.remove(segment_file_name)

        if os.path.exists("temp"):
            shutil.rmtree("temp")


if __name__ == "__main__":
    face_image = Image.from_path("example.jpeg")
    inp = FaceswapInput(face_image=face_image, output_video_type="short")
    model = FaceswapRoop()
    model.setup()
    output = model.predict([inp])[0]
    print(output.output_video)
