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

On Windows Portable, run the installer from the
`ComfyUI_windows_portable` directory with ComfyUI's embedded Python:

```bat
.\python_embeded\python.exe .\ComfyUI\custom_nodes\ComfyUI-Qwen3-TTS\install.py
```

On **ComfyUI Desktop** there is no `python_embeded`. Run the installer with the
same Python that ComfyUI itself uses to run workflows, so the isolated runtime
reuses ComfyUI's CUDA build of Torch. The Desktop venv lives next to the
install, and its exact path is shown on startup as `Python executable`:

```powershell
& "C:\Users\<you>\Documents\ComfyUI\.venv\Scripts\python.exe" `
  .\ComfyUI-Qwen3-TTS\install.py
```

Do **not** run `install.py` with a separate system Python (for example a
standalone `python install.py`). The installer reuses the Torch of whatever
interpreter launches it, so a system Python builds a CPU-only runtime that
cannot run Qwen3-TTS. `install.py` now checks this and stops with a clear error
if the reused Torch has no CUDA. If you already built a wrong `.venv`, delete
`ComfyUI-Qwen3-TTS\.venv` (the custom-node one, not ComfyUI's own) and re-run
with the correct interpreter.

The installer uses `.venv\Scripts\python.exe` on Windows and
`.venv/bin/python` on Linux/macOS.
If the embedded Windows Python cannot create a standard venv, the installer can
fall back to an existing `uv` executable. A `.venv` copied from another OS or
CPU architecture must not be reused; run `install.py` on each target machine.

ComfyUI Desktop and some current builds no longer bundle Torchaudio, which
`qwen-tts` imports at load time. When the reused environment has no Torchaudio,
`install.py` installs a build that matches the reused Torch's exact version and
CUDA suffix (for example `torchaudio==2.10.0+cu130` for `torch 2.10.0+cu130`),
without pulling a second Torch or CUDA stack.

`qwen-tts 0.1.1` pins Transformers 4.57.3, which conflicts with current ComfyUI
releases using Transformers 5.x. `install.py` therefore creates
`ComfyUI-Qwen3-TTS/.venv` and installs the official Qwen runtime there. This
small dependency layer reuses ComfyUI's existing Torch, TorchAudio, and CUDA
packages instead of installing a second GPU stack. The nodes still execute
synchronously inside ComfyUI's normal prompt queue, while ComfyUI's own Python
packages remain unchanged.

After installing the isolated runtime, `install.py` automatically downloads
both official models into `ComfyUI/models/qwen-voice/`. New downloads use a
portable direct-file layout and therefore do not require Windows symlink
privileges. Existing complete Hugging Face caches are detected and reused, so
upgrading an existing N5, Spark, or Windows installation does not download a
second copy. Set `QWEN3_TTS_SKIP_MODEL_DOWNLOAD=1` to install dependencies only,
or `QWEN3_TTS_MODEL_ROOT` to override the model directory.

While a prompt is running, the nodes publish standard ComfyUI progress events
for queueing, model loading, reference preparation, generation, validation, and
output saving. Generation advances from 35% to 92% using a conservative runtime
estimate; it is an ETA-based progress indication, not a token-level callback
from Qwen3-TTS.

## Models

`install.py` downloads both official model repositories automatically. New
installations use:

```text
ComfyUI/models/qwen-voice/Qwen3-TTS-12Hz-1.7B-VoiceDesign/
ComfyUI/models/qwen-voice/Qwen3-TTS-12Hz-1.7B-Base/
```

Required models:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

Existing installations using the following Hugging Face cache layout remain
fully supported and are reused without another download:

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
