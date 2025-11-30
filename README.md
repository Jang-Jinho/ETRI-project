# ETRI Project

멀티모달 AI 기반 미디어 핵심 정보 분석 및 요약 기술 (2025/07/16 ~ 2025/11/30)

## Environment Setup
  LongVALE: https://github.com/ttgeng233/LongVALE
```bash
# Environment 1 for LongVALE
# 1-1. Tree Construct
# 1-2. Leaf Node Captioning  
conda create --name eventtree python=3.10
conda activate eventtree
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
pip install soundfile
pip install streamlit
```

  LLaMA3: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```bash
# Environment 2 for LLaMA3
# 2-1. Internal Node Captioning
# 2-2. Structured Data Postprocessing
conda create --name eventtree-post python=3.10
conda activate eventtree-post
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install cuda-toolkit==12.8.1
pip install transformers accelerate peft
pip install decord
pip install jsonschema
pip install sentence-transformers
```

```bash
# System package
# If it's not installed, install it
sudo apt update
sudo apt install -y ffmpeg yt-dlp
(conda install -c conda-forge ffmpeg yt-dlp)
```

## Data Setup
- `annotation.json`, `prompt.json`, (`video.mp4`, `audio.wav`) are required.
- `annotation.json`: Must include the video id (YouTube id) and duration.
- `prompt.json`: Contains the prompt input during the Node Captioning process.
- `audio.wav`: Audio must be extracted from `video.mp4` (using ffmpeg).
```shell
# Tree Feature Extraction (features_tree)
# Type: all, video, audio, speech
bash scripts/features_tree.sh <TYPE>
```

```shell
# LongVALE Feature Extraction (features_eval)
# Type: all, video, audio, speech, speech_asr
bash scripts/features_longvale.sh <TYPE>
```

`data` directory is as follows:

```text
data/
├── annotation.json
├── prompt.json
├── raw_data/
│   ├── video/{video_id}.mp4 # input
│   └── audio/{video_id}.wav # input
├── features_tree/
│   ├── video_features/{video_id}.npy
│   ├── audio_features/{video_id}.npy
│   └── speech_features/{video_id}.npy
└── features_model/
    ├── video_features/{video_id}.npy
    ├── audio_features/{video_id}.npy
    ├── speech_features/{video_id}.npy
    └── speech_asr/{video_id}.json
```
    
## Checkpoint Setup

| Modality      | Encoder | Checkpoint path                           | Download checkpoint                                                                 |
|---------------|---------|-------------------------------------------|-------------------------------------------------------------------------------------|
| Visual        | CLIP    | `./checkpoints/ViT-L-14.pt`               | [ViT-L/14](https://huggingface.co/datasets/ttgeng233/LongVALE/resolve/main/checkpoints/ViT-L-14.pt?download=true)                                         |
| Audio         | BEATs   | `./checkpoints/BEATs_iter3_plus_AS20K.pt` | [BEATs_iter3_plus_AS20K](https://huggingface.co/datasets/ttgeng233/LongVALE/resolve/main/checkpoints/BEATs_iter3_plus_AS20K.pt?download=true)     |
| Speech        | Whisper | `./checkpoints/openai-whisper-large-v2`   | [whisper-large-v2](https://huggingface.co/openai/whisper-large-v2)                 |

- LongVALE: Download [Vicuna v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) and [vtimellm_stage1](https://huggingface.co/datasets/ttgeng233/LongVALE/blob/main/checkpoints/vtimellm_stage1_mm_projector.bin) weights.
- LongVALE: Download LongVALE-LLM model from [longvalellm-vicuna-v1-5-7b.tar.gz](https://huggingface.co/datasets/ttgeng233/LongVALE/resolve/main/checkpoints/longvalellm-vicuna-v1-5-7b.tar.gz).
  
`checkpoints` directory is as follows:

```text
checkpoints/
├── ViT-L-14.pt
├── BEATs_iter3_plus_AS20K.pt
├── openai-whisper-large-v2 # folder 
├── vicuna-7b-v1.5 # folder
├── vtimellm_stage1_mm_projector.bin 
└── longvalellm-vicuna-v1-5-7b # folder
    ├── longvale-vicuna-v1-5-7b-stage2-bp
    └── longvale-vicuna-v1-5-7b-stage3-it
```

## How to Run
### **Main**
```shell
bash scripts/run.sh
```

### **Demo 1**
Input: Video file (.mp4)
```shell
bash scripts/run_demo.sh <VIDEO_ID> <QUERY>
```
The data/output paths used in `scripts/run_demo.sh` are as follows:

```text
data/
└── prompt.json

demo/
├── {video_id}.mp4 # run_demo.sh (input)
│ 
├── outputs/
│   ├── log.json # Tree (Caption) result
│   ├── {video_id}.json # Tree (Structured Data) result 
└── └── query/ 
        └── {video_id}.json # Query result
```

### **Demo 2**
Input: Video link (URL)
```shell
bash scripts/run_demo_url.sh <VIDEO_LINK> <QUERY>
```
The data/output paths used in `scripts/run_demo_url.sh` are as follows:

```text
data/
└── prompt.json

demo/
├── outputs/
│   ├── log.json # Tree (Caption) result
│   ├── demo.json # Tree (Structured Data) result 
└── └── query/ 
        └── demo.json # Query result
```

## Example Result
- Event Captions for each node are organized into a multi-level tree structure (Visual-Audio-Speech).
- Multi-level captions are leveraged to generate structured metadata (Tags, Objects, Actors, etc.).
- The generated metadata enables efficient and precise query search and information retrieval.
  
![sample figure](asset/sample.png)
