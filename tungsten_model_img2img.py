import os
import warnings
from typing import List

import torch  # torch should be imported before insightface to use GPUs
from tungstenkit import BaseIO, Field, Image, define_model

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# single thread doubles cuda performance - needs to be set before torch import
os.environ["OMP_NUM_THREADS"] = "1"
# reduce tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import roop.globals
import roop.metadata
from roop.core import (
    limit_resources,
    normalize_output_path,
    pre_check,
    start,
    suggest_execution_threads,
)
from roop.processors.frame.core import get_frame_processors_modules

OUTPUT_FILE_NAME = "./output.png"


class FaceswapInput(BaseIO):
    face_image: Image = Field(description="Input face image")
    target_image: Image = Field(description="Target for swapping a face")


class FaceswapOutput(BaseIO):
    output_image: Image


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
        roop.globals.many_faces = True
        roop.globals.reference_face_position = 0
        roop.globals.similar_face_distance = 0.85
        roop.globals.max_memory = None
        roop.globals.execution_providers = ["CUDAExecutionProvider"]
        roop.globals.execution_threads = suggest_execution_threads()
        if not pre_check():
            return

        self.frame_processors = get_frame_processors_modules(
            ["face_swapper", "face_enhancer"]
        )
        for frame_processor in self.frame_processors:
            if not frame_processor.pre_start():
                return
        limit_resources()

    def predict(self, inputs: List[FaceswapInput]) -> List[FaceswapOutput]:
        input = inputs[0]

        roop.globals.source_path = str(input.face_image.path)
        roop.globals.target_path = str(input.target_image.path)
        roop.globals.output_path = normalize_output_path(
            roop.globals.source_path, roop.globals.target_path, OUTPUT_FILE_NAME
        )

        # shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        # for frame_processor in self.frame_processors:
        #     frame_processor.process_image(
        #         roop.globals.source_path,
        #         OUTPUT_FILE_NAME,
        #         OUTPUT_FILE_NAME,
        #     )
        #     frame_processor.post_process()
        start()

        return [FaceswapOutput(output_image=Image.from_path(OUTPUT_FILE_NAME))]


if __name__ == "__main__":
    face_image = Image.from_path("example.jpeg")
    inp = FaceswapInput(
        face_image=face_image, target_image=Image.from_path("doggy.jpg")
    )
    model = FaceswapRoop()
    model.setup()
    output = model.predict([inp])[0]
