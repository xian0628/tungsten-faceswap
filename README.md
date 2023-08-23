# Tungsten faceswap 
Take a video and replace the face in it with a face of your choice. You only need one image of the desired face. No dataset, no training.

## Prerequisites

- Video clips
- [Python 3.9](https://www.python.org/downloads/release/python-3913/)
- [Docker](https://docs.docker.com/get-docker/)
- (Optional) [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker) for locally testing the model.


## Prepare video clips
Copy your video clips on the project root directory.  
Then, update `tungsten_model.py` corresponding to your needs:
1. Change `FaceswapInput` class - This class is for defining inputs.
2. Change `VIDEO_SEGMENTS` variable - Change file names to your owns.
3. Change `REFERENCE_FRAME_NUMBERS` variable - Change frame number containing the target face.

## Build, push and run your model

### Step 0: Install Requirements

First, install requirements:

```bash
pip install -r requirements.txt
```

### Step 1. Prepare weights

Download weights:
```
./download_weights.sh
```

### Step 2. Build model

```bash
tungsten build . faceswap:myversion
```

### Stel 3 (Optional) Test locally

```bash
tungsten demo
```

### Step 3: Create a project on Tungsten

Go to [tungsten.run](https://tungsten.run/new) and create a project.

### Step 4: Push the model to Tungsten

Log in to Tungsten:

```bash
tungsten login
```

Add tag of the model:
```bash
tungsten tag faceswap:myversion <project_name>
```

Push the model to the project:
```bash
tungsten push <project_name>
```

### Step 5: Run the model on Tungsten

Visit [tungsten.run](https://tungsten.run) and go to the project page.
