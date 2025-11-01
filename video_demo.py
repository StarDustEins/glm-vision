import base64
import os
import tempfile
from typing import Tuple

from moviepy import VideoFileClip

from openai import OpenAI

VIDEO_PATH = (
    "/Users/elysioneins/glm-vision/video_sample/微信视频2025-11-01_101811_282.mp4"
)
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

client = OpenAI(
    api_key=os.getenv(
        "SILICONFLOW_API_KEY",
        "sk-khdvgcgwzbhwqhfcrlbaoazfegxfypunnlgpnwkhbydqqvth",
    ),
    base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
)


def extract_audio_to_base64(path: str, sample_rate: int = 16_000) -> Tuple[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    print(f"Extracting audio track from: {path}")
    with VideoFileClip(path) as clip:
        if clip.audio is None:
            raise ValueError("视频缺少可用的音频轨道，无法执行语音解析。")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_audio_path = tmp.name

        try:
            clip.audio.write_audiofile(
                temp_audio_path,
                codec="pcm_s16le",
                fps=sample_rate,
                # verbose=False,
                logger=None,
            )
            with open(temp_audio_path, "rb") as audio_file:
                encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
        finally:
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass

    return f"data:audio/wav;base64,{encoded_audio}", "wav"


audio_base64, audio_format = extract_audio_to_base64(VIDEO_PATH)

completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
仔细解析视频中的对话内容，并转换成文字。
 - 注意说话人的音色、音调，以区分不同发言人
 - 以A、B、C等英文字符来标记不同发言人
 - 参照样例，以如下json形式返回

```json 
[
    {
        "start_time": "00:00:00.000",
        "end_time": "00:00:05.000",
        "speaker": "A",
        "text": "你好啊，今儿个这天气不错吧。"
    },
    {
        "start_time": "00:00:05.000",
        "end_time": "00:00:07.000",
        "speaker": "B",
        "text": "是的，阳光明媚，心情也跟着好了起来。"
    }
]
```
""",
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": audio_base64,
                    },
                },
            ],
        },
    ],
    modalities=["text", "audio"],
    audio={"voice": "Cherry", "format": "wav"},
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in completion:
    if chunk.choices:
        print(chunk.choices[0].delta)
    else:
        print(chunk.usage)
