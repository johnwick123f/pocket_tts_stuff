
# Character creation information

Custom voice synthesis via `velvet.generate_voice(description=description, **args)`.

| Parameter | Range / Options | Description |
| :--- | :--- | :--- |
| `speed` | `0.5` – `1.5` | Speech rate (higher is faster). |
| `pitch` | `-5.0` – `5.0` | Vocal pitch (higher is higher-pitched). |
| `emotion` | `calm`, `happy`, `sad`, `angry`, `fearful`, `surprise`, `disgust` | emotional tone. |
| `emotion_strength` | `0.0` – `2.0` | Emotion intensity (`~0.5` normal; `1–2` extreme, may affect timbre). |
| `gender` | `male`, `female` | Target vocal gender. |
| `gender_strength` | `0.0` – `1.0` | Gender adherence (`1.0` for strict; lower for blended/neutral). |
| `description` | `string` | Should be overall character name/information, not specific details. |
