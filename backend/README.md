# MedSAM2 FastAPI Backend

A FastAPI-based backend for medical video segmentation using MedSAM2. This application provides RESTful APIs for uploading surgical videos, performing interactive segmentation with point clicks, and generating segmented video outputs.

## Features

- **Video Upload & Processing**: Upload surgical videos with optional time trimming and downsampling
- **Interactive Segmentation**: Click-based point annotation for object segmentation
- **Video Propagation**: Automatic mask propagation across video frames
- **Output Generation**: Generate segmented videos with colored overlays
- **Session Management**: Multi-session support with automatic cleanup

## Prerequisites

- **GPU Requirements**: NVIDIA GPU with CUDA support (2 GB VRAM for 1280x720p@30fps for 10 seconds)
- **Python**: 3.8 or higher
- **FFmpeg**: Required for video processing
- **CUDA Toolkit**: Compatible with your PyTorch installation

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:Rishcursion/Univlabs-Tracker.git && cd backend 
```

### 2. Clone MedSAM2 Repository 

```bash
git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2
# Install Checkpoints
sudo chmod +x download.sh
./download.sh
```

### 3. Install Dependencies

```bash
pip install -e .
pip install 'fastapi[standard]'
```

### 4. Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```
**Arch:**
```bash
sudo pacman -Syu
sudo pacman -S ffmpeg 
```
**Windows:**
- Download from [FFmpeg official website](https://ffmpeg.org/download.html)
- Add to PATH environment variable

**macOS:**
```bash
brew install ffmpeg
```

**Note**: Check the [MedSAM2 repository](https://github.com/bowang-lab/MedSAM2) for the latest checkpoint download instructions.

### 6. Verify CUDA Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
```

## Configuration

### Model Paths

Update the following paths in your main application file if needed:

```python
checkpoint = "./MedSAM2/checkpoints/MedSAM2_latest.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t512.yaml"
```

### GPU Memory Optimization

For GPUs with limited VRAM (like MX450 with 2GB):

1. **Reduce video resolution**: Use higher `downsample_factor` values
2. **Limit clip length**: Process shorter segments (10-30 seconds)
3. **Adjust batch processing**: Process fewer frames simultaneously

## Usage

### 1. Start the Server

```bash
fastapi dev inference_medsam2.py
```

The server will start on `http://localhost:8000`, or you can change the port as desired.

### 2. API Documentation

Once running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 3. API Endpoints
**Refer To inference_medsam2.py for pydantic model definitions to ensure compatibility with the below API**
#### Upload Video and Create Session
```http
POST /create_session_upload/
```
- Upload video file
- Optional parameters: `start_time`, `end_time`, `downsample_factor`

#### Add Annotation Points
```http
POST /add_new_points/
```
- Add interactive points for segmentation
- Supports multiple objects and label types

#### Propagate Masks
```http
POST /propagate_in_video
```
- Propagate segmentation across all video frames

#### Generate Output Video
```http
POST /generate_video
```
- Generate final segmented video with overlays

#### Delete Session
```http
DELETE /delete_session/{session_id}
```
- Clean up session data and temporary files

## Performance Optimization

### For Limited VRAM (2-4GB)

```python
# Example parameters for MX450 (2GB VRAM)
downsample_factor = 4  # Reduce resolution by 4x
max_clip_length = "00:00:15"  # 30-second clips
```

### For High-End GPUs (16GB+)

```python
# Example parameters for RTX 4080/4090
downsample_factor = 1  # No downsampling
max_clip_length = "00:10:00"  # 10-minute clips
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Increase `downsample_factor`
   - Process shorter video clips
   - Reduce batch size in model configuration

2. **FFmpeg Not Found**
   - Ensure FFmpeg is installed and in PATH
   - Test with: `ffmpeg -version`

3. **Model Loading Failed**
   - Check checkpoint file exists and is not corrupted
   - Verify model configuration path
   - Ensure CUDA drivers are properly installed

4. **Slow Processing**
   - Consider upgrading GPU for better performance
   - Use SSD storage for faster I/O
   - Optimize video preprocessing parameters

### Performance Benchmarks

| GPU | VRAM | 10-sec clip (198 frames) | 30-min estimated |
|-----|------|--------------------------|------------------|
| MX450(Tested On) | 2GB | 48 seconds | ~3 hours |
| RTX 4080* | 16GB | ~12 seconds | ~45 minutes |
| RTX 4090* | 24GB | ~8 seconds | ~30 minutes |


\* Not tested on, only indicative performance.
## Directory Structure

```
.
├── assets
├── MedSAM2
│   ├── examples
│   ├── notebooks
│   ├── sam2
│   │   ├── configs
│   │   ├── csrc
│   │   ├── modeling
│   │   │   ├── backbones
│   │   │   └── sam
│   │   └── utils
│   └── training
│       ├── assets
│       ├── dataset
│       ├── model
│       ├── scripts
│       └── utils
├── __pycache__
├── sam2
│   ├── configs
│   │   ├── sam2
│   │   ├── sam2.1
│   │   └── sam2.1_training
│   ├── csrc
│   ├── modeling
│   │   ├── backbones
│   │   └── sam
│   └── utils
├── sav_dataset
│   ├── example
│   └── utils
├── tools
└── training
    ├── assets
    ├── dataset
    ├── model
    ├── scripts
    └── utils

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project follows the licensing terms of the MedSAM2 and SAM2 project. Please refer to the [MedSAM2 repository](https://github.com/bowang-lab/MedSAM2) for license details.

## Acknowledgments

- [MedSAM2](https://github.com/bowang-lab/MedSAM2) - Medical image segmentation model
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Base segmentation architecture
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

## Support

For issues related to:
- **MedSAM2 model**: Check the [official repository](https://github.com/bowang-lab/MedSAM2)
- **API implementation**: Create an issue in this repository
- **Performance optimization**: See the troubleshooting section above
