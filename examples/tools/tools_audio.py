#!/usr/bin/env python3

"""
Example demonstrating text-to-speech and speech-to-text with AudioToolkit.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Add the parent directory to sys.path to import from evoagentx
sys.path.append(str(Path(__file__).parent.parent))

from evoagentx.tools import AudioToolkit
from evoagentx.tools.tool import ToolResult

TEST_AUDIO_URL = "https://xrlxdubfbtizvlyipchg.supabase.co/storage/v1/object/public/howone/projects/rp6jerj/tests/test_audio.mp3"


def unwrap_result(result):
    if isinstance(result, ToolResult):
        return result.result
    return result


def format_tool_result(result):
    if isinstance(result, ToolResult):
        return result.to_dict()
    return result


async def run_tts_stt_showcase():
    print("\n===== AUDIO TOOLKIT TTS + STT SHOWCASE =====\n")

    toolkit = AudioToolkit(
        name="DemoAudioToolkit",
        volcengine_appid=os.getenv("VOLCENGINE_APPID"),
        volcengine_app_access_token=os.getenv("VOLCENGINE_APP_ACCESS_TOKEN"),
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        save_path="./audio_files/examples",
    )

    text = "Welcome to EvoAgentX audio tools. This short clip will be transcribed next."
    tts = toolkit.get_tool("generate_audio_from_text")
    stt = toolkit.get_tool("recognize_audio_to_text")

    print("Generating audio...")
    tts_tool_result = await tts(
        text_to_generate=text,
        languages=["en-US"],
        gender="female",
        audio_output={
            "audio_name": "evoagentx_audio_showcase",
            "audio_format": "mp3",
        },
    )
    print("tts_result: ", format_tool_result(tts_tool_result))
    tts_result = unwrap_result(tts_tool_result)

    if not tts_result.get("success"):
        print(f"TTS failed: {tts_result.get('error', 'Unknown error')}")
        return

    audio_url = tts_result["audio_url"]
    print(f"Generated audio: {audio_url}")

    print("Recognizing test audio...")
    stt_tool_result = await stt(
        audio_url=TEST_AUDIO_URL,
        audio_format="mp3",
        language="en-US",
    )
    print("stt_result: ", format_tool_result(stt_tool_result))
    stt_result = unwrap_result(stt_tool_result)

    if not stt_result.get("success"):
        print(f"STT failed: {stt_result.get('error', 'Unknown error')}")
        return

    print("Transcription:")
    print(stt_result.get("text", ""))


if __name__ == "__main__":
    asyncio.run(run_tts_stt_showcase())
