# VelvetVoice

VelvetVoice is a lightweight, high-speed Text-to-Speech model built on a 1-step flow-matching architecture. It supports voice cloning, text-prompted voice generation, complex text normalization, and low-latency streaming.



## Main Features

- Fast & Lightweight: Runs at 2x–20x realtime on CPU (60x–800x on GPU) in ~250 MB to <2 GB RAM.
- Flexible Voice Control: Supports voice cloning, preset characters, and text-prompted voices (emotion, pitch, timbre, speed).
- Neural Text Normalization: Converts complex symbols and formatting into natural speech.
## Installation

Install VelvetVoice directly from GitHub:

```bash
pip install git+https://github.com/johnwick123f/pocket_tts_stuff.git
```



## Quick Start

#### 1. Load the Model

```python
from velvet_tts import VelvetVoice

velvet = VelvetVoice(
    bundle="small",  # "small" or "mini"
    device="cpu",    # "cpu" or "cuda"
    flow_gain=1.0,
)
```

#### 2. Preset Character Generation

Check information/characters.txt for the full list of built-in voices.

```python
text = "Hey, this is VelvetVoice!"
voice_name = "anna"

audio = velvet.gen(text, voice=voice_name, output_path="output.wav")
```

#### 3. Text-Prompted Voice Generation

Check information/character_creation.txt for details on crafting voice prompts.

```python
text = "Hey, this is a scary zombie."
description = "A scary dangerous zombie voice."

control_voice = velvet.generate_voice(description=description, speed=1.0)
audio = velvet.gen(text, voice=control_voice, output_path="output_zombie.wav")
```

#### 4. Voice Cloning

```python
text = "Hey, this is a cool character!"
voice_path = "clone.wav"

audio = velvet.gen(text, voice=voice_path, output_path="output_cloned.wav")
```

#### 5. Complex Text Normalization

Works with any of the voice modes above.

```python
text = "For x ∈ [-3, 5), |2x - 1/2| ≤ 10.5 when x ≠ 0."
voice_name = "anna"

normalized_text = velvet.normalize(text)
audio = velvet.gen(normalized_text, voice=voice_name, output_path="output_complex.wav")
```

#### 6. Low-Latency Streaming

Works with any of the voice modes above.

```python
prompt = "Hello. The stars look beautiful tonight."
voice = "anna"

for audio_chunk in velvet.stream(prompt, voice=voice, chunk_frames=4):
    print(audio_chunk.shape, velvet.stream_sample_rate)  # Always 24000 Hz
```

## Tips
- Please use at minimum a 3 second audio file for voice cloning.
- Check out /information folder. There are lots of information about control.
- To improve speed and quality if you are using GPU, try utilizing batch feature. Significantly increases speed. 


## Other Info

Q: When should I use ```velvet.normalize()```?

A: For standard sentences, you don't need to. However complex sentences with symbols, different notations, etc., normalize will significantly increase accuracy and quality.

Q: Does it require an internet connection after installation?

A: No. Once the model weights are downloaded on the first run, everything runs 100% locally and offline on your machine.

Q: Can it use less memory?

A: Although VelvetVoice is already very efficient, there are several parts to optimize that can reduce memory usage significantly. Should come later.

## Roadmap

- [x] Release model and code
- [ ] Colab Notebook for simple usage
- [ ] Huggingface spaces demo
- [ ] Optimize code further
- [ ] Stronger and more controls
      
# Acknowledgements
- [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts.git) for their excellent code and model.
- [MioCodec](https://github.com/Aratako/MioCodec.git) for their great codec.
- [Granite-R2-small](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) for their fantastic retrieval model.
- [MNN-Pocket](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B) for their wonderful small LM.
  
## Final Notes

The model and code are licensed under the CC-BY-4.0 license permitting research, personal, and commercial uses given attribution. See LICENSE for details.

Stars/Likes would be appreciated, thank you.

Email: yatharthsharma3501@gmail.com
