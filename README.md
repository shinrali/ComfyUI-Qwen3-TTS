# ComfyUI-Qwen3-TTS

Native ComfyUI workflow nodes for Qwen3-TTS Voice Design and speaker-embedding
Voice Clone / TTS. Generation runs inside ComfyUI's ordinary execution queue, so
it is serialized with image, video, and other audio workflows on the same GPU.

## Nodes

- `Qwen3-TTS Voice Design`
- `Qwen3-TTS Voice Clone / TTS`
- `Save Qwen3-TTS Audio`

The clone node uses `x_vector_only_mode=True`; it derives speaker identity from
the reference audio and does not require a reference transcript. Models are
released after each workflow so ComfyUI can reclaim GPU memory for the next job.

## Install

Clone this repository into the normal ComfyUI custom node directory, run its
installer once, and restart ComfyUI. No custom image, Dockerfile, service, port,
or API is required.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/shinrali/ComfyUI-Qwen3-TTS.git
python ComfyUI-Qwen3-TTS/install.py
```

On Windows Portable, run the installer with ComfyUI's embedded Python:

```bat
..\python_embeded\python.exe ComfyUI-Qwen3-TTS\install.py
```

Run that command from the `ComfyUI\custom_nodes` directory. The installer uses
`.venv\Scripts\python.exe` on Windows and `.venv/bin/python` on Linux/macOS.
If the embedded Windows Python cannot create a standard venv, the installer can
fall back to an existing `uv` executable. A `.venv` copied from another OS or
CPU architecture must not be reused; run `install.py` on each target machine.

`qwen-tts 0.1.1` pins Transformers 4.57.3, which conflicts with current ComfyUI
releases using Transformers 5.x. `install.py` therefore creates
`ComfyUI-Qwen3-TTS/.venv` and installs the official Qwen runtime there. This
small dependency layer reuses ComfyUI's existing Torch, TorchAudio, and CUDA
packages instead of installing a second GPU stack. The nodes still execute
synchronously inside ComfyUI's normal prompt queue, while ComfyUI's own Python
packages remain unchanged.

While a prompt is running, the nodes publish standard ComfyUI progress events
for queueing, model loading, reference preparation, generation, validation, and
output saving. Generation advances from 35% to 92% using a conservative runtime
estimate; it is an ETA-based progress indication, not a token-level callback
from Qwen3-TTS.

## Models

Download both official model repositories into a Hugging Face cache rooted at:

```text
ComfyUI/models/qwen-voice/huggingface/
```

Required models:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

The resulting layout includes:

```text
ComfyUI/models/qwen-voice/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign/
ComfyUI/models/qwen-voice/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/
```

Inference is local-only and does not download missing weights during a queued
workflow. Missing models produce a clear node error.

## Example workflows

Drag either file from `example_workflows/` into ComfyUI:

- `Qwen3-TTS Voice Design.json`
- `Qwen3-TTS Voice Clone TTS.json`

Voice Design creates one or more candidates from text and a voice instruction.
Voice Clone / TTS accepts a normal ComfyUI `LoadAudio` reference.

## Notes

- NVIDIA CUDA with BF16 support is currently expected. Linux and Windows are
  supported; CPU and non-CUDA Windows builds are not currently supported.
- FlashAttention 2 is used when already available; otherwise the node uses SDPA.
- Long text is rejected by a conservative duration guard and should be split
  into narration segments.
- Voice cloning must only be used with the speaker's permission and applicable
  rights.

## License

MIT
