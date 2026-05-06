# config files dependencies: evoagentx/tools/audio_tools/assets/volcengine_voices.json
from typing import Dict, Optional, List, Any
import os
import copy
import uuid
import json
import time
import asyncio
import io
import wave
from urllib.parse import urlparse

import httpx

try:
    import av
except ImportError:
    av = None

try:
    import websockets
except ImportError:
    websockets = None

from ..tool import Tool, ToolResult, ToolMetadata, Toolkit
from ..storage_handler import FileStorageHandler, LocalStorageHandler, SupabaseStorageHandler
try:
    from .volcengine_utils.protocols.protocols import full_client_request, receive_message, MsgType
except ImportError:
    full_client_request = None
    receive_message = None
    MsgType = None
# from evoagentx.utils.volcengine_utils.volc_tts_params_info import PARAM_VOICE_TYPE_DESC, PARAM_EMOTION_DESC
from evoagentx.core.logging import logger
from evoagentx.core.module import BaseModule
from evoagentx.core.module_utils import parse_json_from_llm_output


class AudioFileMetaData(BaseModule):
    size_bytes: int
    mime_type: str
    language: Optional[List[str]] = None
    duration_seconds: Optional[float] = None
    sample_rate_hz: Optional[int] = None
    channels: Optional[int] = None

VOLCENGINE_TTS_PRICE = {
    "post_paid": {
        "price":5/7, # cost in dollars, fix rate, to be updated
        "unit": 10000
    }
}

VOLCENGINE_STT_PRICE = {
    "post_paid": {
        "price": 1.8/7,
        "unit": 3600000  # in ms, one hour
    }
}

AUDIO_LLM_MODEL="google/gemini-2.5-flash"

AUDIO_GENERATION_EXTRA_DESC = """
Example 1: Basic text-to-speech generation with default settings
Arguments:
{
    "text_to_generate": "Welcome to our podcast. Today we will explore the fascinating world of artificial intelligence.",
    "audio_output": {
        "audio_name": "podcast_intro",
        "audio_format": "mp3"
    }
}

Returns a dict with audio URL and metadata:
{
    "success": true,
    "audio_url": "https://xxx.supabase.co/xxx/podcast_intro.mp3",
    "audio_metadata": {
        "size_bytes": 245760,
        "mime_type": "audio/mp3",
        "language": ["en-US"],
        "duration_seconds": 5.2,
        "sample_rate_hz": 24000,
        "channels": 1
    },
    "resolved_voice_id": "en_female_lauren_moon_bigtts",
    "voice_name": null,
    "emotion": null,
    "selection_method": "default"
}

Example 2: Customized voice generation with language, gender, and audio_hint
Use `languages` to specify the language, `gender` to choose male/female voice, and `audio_hint` to describe desired voice characteristics (role, emotion, tone).
Arguments:
{
    "text_to_generate": "亲爱的顾客您好，欢迎致电客服中心，我是您的专属客服小美，很高兴为您服务。",
    "languages": ["zh-CN"],
    "gender": "female",
    "audio_hint": "Female voice, warm and friendly, like a professional customer service representative, gentle and patient tone",
    "audio_output": {
        "audio_name": "customer_service_greeting",
        "audio_format": "wav"
    }
}   


Returns a customized voice audio with LLM-selected voice characteristics:
{
    "success": true,
    "audio_url": "https://xxx.supabase.co/storage/v1/object/public/xxx/customer_service_greeting.wav",
    "audio_metadata": {
        "size_bytes": 512000,
        "mime_type": "audio/wav",
        "duration_seconds": 6.8,
        "sample_rate_hz": 24000,
        "channels": 1,
        "language": ["zh-CN"]
    },
    "resolved_voice_id": "zh_female_cancan_mars_bigtts",
    "voice_name": "灿灿",
    "emotion": "gentle",
    "selection_method": "llm_selected",
    "llm_reason": "Selected this voice for its warm and friendly tone, suitable for customer service scenarios"
}
"""

AUDIO_RECOGNITION_EXTRA_DESC = """
Example 1: Basic speech-to-text transcription with default settings
Arguments:
{
    "audio_url": "https://storage.example.com/audio/meeting_recording.mp3",
    "audio_format": "mp3"
}

Returns a dict with transcribed text:
{
    "success": true,
    "text": "Hello everyone, welcome to today's meeting. Let's start by reviewing the agenda for this session."
}

Example 2: Multi-speaker transcription with language specification and ITN
Use `language` to specify the audio language for better accuracy, `with_speaker_info=true` to identify different speakers, and `use_itn=true` to normalize numbers and dates.
Arguments:
{
    "audio_url": "https://storage.example.com/audio/interview_chinese.wav",
    "audio_format": "wav",
    "language": "zh-CN",
    "use_itn": true,
    "with_speaker_info": true
}

Returns transcribed text with speaker diarization and normalized numbers:
{
    "success": true,
    "text": "主持人：欢迎来到今天的访谈节目，今天是2024年3月15日。嘉宾：谢谢邀请，很高兴能参加这次节目。主持人：请问您对人工智能的发展有什么看法？"
}

NOTE: For accurate multilingual recognition, it is recommended to specify the `language` parameter whenever possible. If not specified, the default language is English (en-US).
"""

VOICE_SELECTION_PROMPT = """You are a voice selection assistant. Based on the user's requirements, select the most appropriate voice from the available candidates.

## User Requirements:
- Languages: {languages}
- Gender: {gender}
- Audio Hint (desired voice characteristics): {audio_hint}

## Available Voice Candidates:
{candidates_info}

## Instructions:
1. Analyze the user's requirements carefully, especially the audio_hint which describes the desired voice characteristics.
2. Match the requirements against the available candidates' hints and characteristics.
3. If the voice supports emotions and the audio_hint suggests a specific emotional tone, select an appropriate emotion.
4. If there is no perfect match among the candidates, select the closest match based on overall similarity (voice characteristics, tone, style, etc.).
5. Output your selection in the following JSON format ONLY (no other text):

```json
{{
    "voice_id": "<the exact id of the selected voice>",
    "voice_name": "<the name of the selected voice>",
    "emotion": "<selected emotion if applicable, otherwise null>",
    "reason": "<brief explanation of why this voice was selected>"
}}
```

Important:
- The voice_id MUST be copied exactly from the candidates list.
- Only select an emotion if the voice supports emotions (support_emotion=true) AND the audio_hint suggests an emotional tone.
- If no emotion is needed or the voice doesn't support emotions, set emotion to null.
- You MUST always select a voice from the candidates, even if none is a perfect match. Choose the best available option.
"""

DEFAULT_VOICE_CONFIG = {
    "zh-CN": {
        "male": "zh_male_chunhui_mars_bigtts",
        "female": "zh_female_cancan_mars_bigtts"
    },
    "en-US": {
        "male": "en_male_michael_moon_bigtts",
        "female": "en_female_lauren_moon_bigtts"
    },
    "en-GB": {
        "male": "en_male_dave_moon_bigtts",
        "female": "en_female_anna_mars_bigtts"
    },
    "en-AU": {
        "male": "ICL_en_male_aussie_v1_tob",
        "female": "en_female_sarah_mars_bigtts"
    },
    "ja-JP": {
        "male": "multi_zh_male_youyoujunzi_moon_bigtts",
        "female": "multi_female_maomao_conversation_wvae_bigtts"
    },
    "es-ES": {
        "male": "multi_male_M100_conversation_wvae_bigtts",
        "female": "multi_female_maomao_conversation_wvae_bigtts"
    }
}

class AudioGenerationTool(Tool):
    name: str = "generate_audio_from_text"
    description: str = (
        "Generate audio from text input with configurable parameters including language (en-US, zh-CN, en-GB, en-AU, ja-JP, es-ES), "
        "gender (male/female), audio_hint (natural language description of voice characteristics like role, emotion, tone), "
        "Supports multilingual audio generation by mixing Chinese and English in the same text. "
        "Returns a dict containing 'success' (bool), 'audio_url' (Supabase/URL link to the generated audio file), "
        "'audio_metadata' (technical details: duration, sample_rate, channels, mime_type), and 'voice_name'/'emotion' (the voice characteristics used). "
        "You should use this tool when dealing with tasks related to text-to-speech generation, audio narration, "
        "voiceover production, podcast creation, or generating voice responses for conversational AI. "
        "NOTE: This tool is designed for generating single-speaker audio. For multi-speaker dialogues, "
        "call this tool separately for each speaker's lines, then use merge_audio_files to combine them. "
        "Ensure voice consistency by using the same audio_hint and gender settings for the same speaker throughout."
    )
    extra_description: str = AUDIO_GENERATION_EXTRA_DESC.strip()
    inputs: Dict = {
        "text_to_generate": {
            "type": "string",
            "description": "Required. The text content to convert into audio speech.",
        },
        "languages": {
            "type": "array",
            "description": (
                "Optional. A list of language codes for the audio generation. "
                "For single-language audio, provide one language code (e.g., ['en-US'] or ['zh-CN']). "
                "For multilingual audio (mixing Chinese and English in the same text), provide both codes (e.g., ['zh-CN', 'en-US']). "
                "NOTE: Multilingual support is only available for Chinese and English. Default is ['en-US']."
            ),
            "items": {
                "type": "string",
                "enum": ["en-US", "zh-CN", "en-GB", "en-AU", "ja-JP", "es-ES"]
            }
        },
        "gender": {
            "type": "string",
            "description": "Optional. The gender of the voice. Default is 'female'.",
            "enum": ["male", "female"]
        },
        "audio_hint": {
            "type": "string",
            "description": (
                "Optional. A natural language description of the desired voice characteristics, following the format: "
                "'[Role] + [Gender] + [Language Style] + [Emotion] + [Scenario]'. "
                "Examples: 'Female voice, gentle and warm, like a kind teacher giving a lecture, calm tone'. "
                "NOTE: audio_hint MUST be written in English regardless of the input text or the targeted audio language."
            ),
        },
        "audio_output": {
            "type": "object",
            "description": "Optional. Output configuration for the generated audio file.",
            "properties": {
                "audio_name": {
                    "type": "string",
                    "description": "The name of the output audio file (without extension). Must use ASCII characters only, regardless of the input text language. If not provided, a default name will be generated."
                },
                "audio_format": {
                    "type": "string",
                    "description": "The output audio format. Supported formats: 'mp3', 'wav'. Default: 'mp3'.",
                    "enum": ["mp3", "wav"]
                }
            }
        }
    }
    required: Optional[List[str]] = ["text_to_generate"]

    def __init__(
        self,
        name: Optional[str] = None,
        openrouter_key: Optional[str] = None,
        volcengine_appid: Optional[str] = None,
        volcengine_app_access_token: Optional[str] = None,
        save_path: str = "./audio_files/volcengine",  # only for local storage(but not actually used)
        tool_user_id: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        endpoint: str = "wss://openspeech.bytedance.com/api/v1/tts/ws_binary",
    ):
        super().__init__()
        self.name = name or self.name
        self.openrouter_key = openrouter_key or os.getenv("OPENROUTER_API_KEY")
        self.volcengine_appid = volcengine_appid or os.getenv("VOLCENGINE_APPID")
        self.volcengine_app_access_token = volcengine_app_access_token or os.getenv("VOLCENGINE_APP_ACCESS_TOKEN")
        self.endpoint = endpoint or os.getenv("VOLCENGINE_TTS_ENDPOINT")
        self.tool_user_id = tool_user_id or str(uuid.uuid4())

        self.check_external_provider_config()

        # storage handler for storing audio files
        if storage_handler is None:
            if os.getenv("SUPABASE_URL_STORAGE", None):
                storage_handler = SupabaseStorageHandler()
            else:
                storage_handler = LocalStorageHandler(base_path=save_path)

        self.storage_handler = storage_handler

        self.task_status_codes_mapping = {
            3000: "Request successful, normal synthesis, processed correctly",
            3001: "Invalid request, some parameter values are illegal (e.g. incorrect operation configuration), please check parameters",
            3003: "Concurrency limit exceeded, online concurrency threshold reached, retry or switch to offline mode when using SDK",
            3005: "Backend service busy, backend server under high load, retry or switch to offline mode when using SDK",
            3006: "Service interrupted, duplicate request with the same reqid after completion or failure, please check parameters",
            3010: "Text length exceeded, single request exceeds the configured text length limit, please check parameters",
            3011: "Invalid text, incorrect parameters or empty text, text-language mismatch, or text contains only punctuation, please check parameters",
            3030: "Processing timeout, single request exceeds the maximum service time limit, retry or check the text",
            3031: "Processing error, backend exception occurred, retry or switch to offline mode when using SDK",
            3032: "Timeout while waiting for audio, backend network exception, retry or switch to offline mode when using SDK",
            3040: "Backend connection error, backend network exception, retry",
            3050: "Voice not found, please check the voice_type identifier and related parameters"
        }
        self.task_status_failed_group = [3001, 3003, 3005, 3006, 3010, 3011, 3030, 3031, 3032, 3040, 3050]
        self.task_status_success_group = [3000]
        self._volcengine_voices = None

        # Concurrency control and timeout settings
        self._websocket_semaphore = asyncio.Semaphore(3)  # Shared semaphore across requests
        self._connection_timeout = 30  # Timeout for websocket connection (seconds)
        self._receive_timeout = 120  # Timeout for receiving audio data (seconds)
        self._max_retries = 3  # Maximum retry attempts for connection failures

        from evoagentx.models.openrouter_model import OpenRouterConfig, OpenRouterLLM
        self.llm = OpenRouterLLM(
            config = OpenRouterConfig(
                openrouter_key=self.openrouter_key,
                model=AUDIO_LLM_MODEL,
                temperature=0.0,
                # stream=True,
                # output_response=True
            )
        )

    @staticmethod
    def get_cluster(voice_type: str) -> str:
        """determine cluster based on voice_type"""
        if voice_type.startswith("S_"):
            return "volcano_icl"
        return "volcano_tts"
    
    def check_external_provider_config(self) -> None:
        if self.volcengine_appid is None:
            raise ValueError("Volcengine appid is required")
        if self.volcengine_app_access_token is None:
            raise ValueError("Volcengine app access token is required")
        if self.endpoint is None:
            raise ValueError("Volcengine TTS endpoint is required")

    def _get_cost(self, text_to_generate: str) -> float:
        """
        cost in dollars
        """
        cost = len(text_to_generate) / VOLCENGINE_TTS_PRICE["post_paid"]["unit"] * VOLCENGINE_TTS_PRICE["post_paid"]["price"]
        return cost

    async def __call__(
        self,
        text_to_generate: str,
        languages: list = None,
        gender: str = None,
        audio_hint: str = None,
        speech_rate: float = 1.0,
        audio_output: dict = None,
        **kwargs
    ) -> ToolResult:
        tool_args = {
            "text_to_generate": text_to_generate,
            "languages": languages,
            "gender": gender,
            "audio_hint": audio_hint,
            "speech_rate": speech_rate,
            "audio_output": audio_output
        }
        metadata = ToolMetadata(tool_name=self.name, args=tool_args)

        # Parse audio output options
        audio_name = audio_output.get("audio_name", f"audio_{uuid.uuid4().hex[:8]}") if audio_output else f"audio_{uuid.uuid4().hex[:8]}"
        if not audio_name:
            return ToolResult(
                result={
                    "success": False,
                    "error": "audio_name cannot be empty. Please provide a valid audio_name in audio_output parameter, or omit it to use auto-generated name.",
                },
                metadata=metadata
            )
        
        audio_format = audio_output.get("audio_format", "mp3") if audio_output else "mp3"
        if audio_format not in ["mp3", "wav"]:
            return ToolResult(
                result={
                    "success": False,
                    "error": f"Unsupported audio_format '{audio_format}'. Supported formats are: ['mp3', 'wav']. Please update audio_output parameter with a valid audio_format value."
                },
                metadata=metadata
            )

        # Determine voice selection approach
        voice_info = {}
        try:
            if not self._needs_custom_voice_selection(languages, gender, audio_hint):
                # Use default voice - no customization needed
                voice_id = self._get_default_voice_id(languages, gender)
                voice_info = {
                    "resolved_voice_id": voice_id,
                    "voice_name": None,
                    "emotion": None,
                    "selection_method": "default"
                }
            else:
                # Custom voice selection via LLM
                # Step 1: Filter candidates by language and gender
                candidates = self._filter_voice_candidates(languages, gender)

                if not candidates:
                    # No matching candidates - fall back to default with relaxed filters
                    # Try without gender filter first
                    candidates = self._filter_voice_candidates(languages, None)
                    if not candidates:
                        # Try without language filter
                        candidates = self._filter_voice_candidates(None, gender)
                    if not candidates:
                        # Use all voices as candidates
                        candidates = self._load_volcengine_voices()

                if len(candidates) == 1:
                    # Only one candidate - use it directly
                    matched_voice = candidates[0]
                    voice_info = {
                        "resolved_voice_id": matched_voice["id"],
                        "voice_name": matched_voice["name"],
                        "emotion": matched_voice.get("default_emotion"),
                        "selection_method": "single_candidate"
                    }
                else:
                    # Step 2: Use LLM to select the best voice
                    llm_selection = await self._select_voice_with_llm(
                        candidates=candidates,
                        languages=languages,
                        gender=gender,
                        audio_hint=audio_hint
                    )
                    logger.info(f"LLM voice selection result: {llm_selection}")
                    
                    # Step 3: Find best matching voice using string similarity
                    llm_voice_id = llm_selection.get("voice_id", "")
                    matched_voice = self._find_best_matching_voice(llm_voice_id, candidates)

                    if matched_voice is None:
                        # LLM selection didn't match any candidate - raise error to fall back to default voice
                        raise RuntimeError(
                            f"LLM selected voice_id '{llm_voice_id}' does not match any candidate voices. "
                            f"Available candidate voice IDs: {[v['id'] for v in candidates[:5]]}"
                        )
                    else:
                        # Validate emotion if selected
                        selected_emotion = llm_selection.get("emotion")
                        if selected_emotion and matched_voice.get("support_emotion"):
                            supported_emotions = matched_voice.get("supported_emotions", [])
                            if selected_emotion not in supported_emotions:
                                # Invalid emotion - use default
                                selected_emotion = matched_voice.get("default_emotion")
                        elif not matched_voice.get("support_emotion"):
                            selected_emotion = None

                        voice_info = {
                            "resolved_voice_id": matched_voice["id"],
                            "voice_name": matched_voice["name"],
                            "emotion": selected_emotion,
                            "selection_method": "llm_selected",
                            "llm_reason": llm_selection.get("reason")
                        }
        
        except Exception as e:
            # Voice selection failed - use default voice
            voice_id = self._get_default_voice_id(languages, gender)
            voice_info = {
                "resolved_voice_id": voice_id,
                "voice_name": None,
                "emotion": None,
                "selection_method": "default_after_error",
                "error": str(e)
            }
        logger.info(f"Voice selection result: {voice_info}")

        # Prepare parameters for audio generation
        voice_id = voice_info["resolved_voice_id"]
        emotion = voice_info.get("emotion")
        enable_emotion = emotion is not None

        # Generate audio
        try:
            audio_data = await self._generate_audio_internal(
                text_to_generate=text_to_generate,
                voice_type=voice_id,
                encoding=audio_format,
                emotion=emotion,
                enable_emotion=enable_emotion,
                speed_ratio=speech_rate
            )
        except Exception as e:
            return ToolResult(
                result={
                    "success": False,
                    "error": f"Error generating audio: {e}",
                    **voice_info
                },
                metadata=metadata
            )

        # Save audio data to storage
        full_audio_name = f"{audio_name}.{audio_format}"
        try:
            save_result = self.storage_handler.save(full_audio_name, audio_data)
            # Check if save operation failed (when success key is explicitly present)
            if "success" in save_result and not save_result["success"]:
                error_msg = save_result.get("error", "Unknown error during save")
                return ToolResult(
                    result={
                        "success": False,
                        "error": f"Error saving audio: {error_msg}",
                        **voice_info
                    },
                    metadata=metadata
                )
        except Exception as e:
            return ToolResult(
                result={
                    "success": False,
                    "error": f"Error saving audio: {e}",
                    **voice_info
                },
                metadata=metadata
            )

        # Get cost
        cost = self._get_cost(text_to_generate)
        metadata.add_cost_breakdown({"volcengine:tts__yu_yin_he_cheng_da_mo_xing_v1": cost})

        # Resolve actual languages used (apply default if not specified)
        resolved_languages = languages if languages else ["en-US"]

        # Get audio metadata
        audio_metadata = self._get_audio_metadata(
            audio_data=audio_data,
            audio_format=audio_format,
            languages=resolved_languages,
        )

        # Build result
        result = {
            "success": True,
            "audio_url": save_result.get("url") or save_result.get("file_path"),
            "audio_metadata": audio_metadata.to_dict(),
            **voice_info
        }

        # Add warning
        parsed_url = urlparse(result["audio_url"])
        file_name = os.path.basename(parsed_url.path)
        basename = os.path.splitext(file_name)[0]
        raw_audio_name = audio_output.get("audio_name") if audio_output else None
        if raw_audio_name and basename != raw_audio_name:
            result["warning"] = f"The saved audio name is different from the requested: requested '{raw_audio_name}', actually saved as '{basename}', use the url in the 'audio_url' field to access the actually saved image!"

        return ToolResult(result=result, metadata=metadata)
    
    async def _receive_audio_data(self, websocket) -> bytes:
        """
        Receive audio data from websocket with timeout protection.

        Args:
            websocket: Active websocket connection

        Returns:
            Audio data as bytes

        Raises:
            RuntimeError: If no audio data received or invalid message type
        """
        audio_data = bytearray()
        while True:
            msg = await receive_message(websocket)

            if msg.type == MsgType.FrontEndResultServer:
                continue
            elif msg.type == MsgType.AudioOnlyServer:
                audio_data.extend(msg.payload)
                if msg.sequence < 0:  # Last message
                    break
            else:
                raise RuntimeError(f"TTS conversion failed: {msg}")

        if not audio_data:
            raise RuntimeError("No audio data received")

        return bytes(audio_data)

    async def _generate_audio_internal(
        self,
        text_to_generate: str,
        voice_type: str,
        encoding: str = "mp3",
        emotion: str = None,
        enable_emotion: bool = False,
        speed_ratio: float = 1.0,
    ) -> bytes:
        """
        Internal method to generate audio using Volcengine TTS API with concurrency control and retry logic.

        Args:
            text_to_generate: The text to convert to speech
            voice_type: The voice ID to use
            encoding: Audio format ('mp3' or 'wav')
            emotion: Emotion to apply (if voice supports it)
            enable_emotion: Whether to enable emotion
            speed_ratio: Speech speed ratio (0.5-2.0)

        Returns:
            Audio data as bytes

        Raises:
            RuntimeError: If connection fails after retries or audio generation fails
            asyncio.TimeoutError: If connection or data receiving times out
        """
        if websockets is None:
            raise ImportError("websockets is required for Volcengine audio generation")
        if full_client_request is None or receive_message is None or MsgType is None:
            raise ImportError("Volcengine websocket protocol helpers are required for audio generation")

        # Use shared semaphore to limit concurrent websocket connections across requests
        async with self._websocket_semaphore:
            # Generate unique user ID per request to avoid conflicts
            unique_user_id = f"{self.tool_user_id}_{uuid.uuid4().hex[:8]}"

            # Prepare request payload
            cluster = self.get_cluster(voice_type)

            base_request = {
                "app": {
                    "appid": self.volcengine_appid,
                    "token": self.volcengine_app_access_token,
                    "cluster": cluster,
                },
                "user": {
                    "uid": unique_user_id,
                },
                "request": {
                    "text": text_to_generate,
                    "operation": "submit",
                    "with_timestamp": "1",
                    "extra_param": json.dumps({
                        "disable_markdown_filter": False,
                    }),
                },
            }

            if enable_emotion:
                base_request["audio"] = {
                    "voice_type": voice_type,
                    "encoding": encoding,
                    "emotion": emotion,
                    "speed_ratio": speed_ratio,
                    "enable_emotion": enable_emotion,
                }
            else:
                base_request["audio"] = {
                    "voice_type": voice_type,
                    "encoding": encoding,
                }

            # Retry logic with exponential backoff
            last_error = None
            for attempt in range(self._max_retries):
                websocket = None
                request_id = str(uuid.uuid4())
                try:
                    # Connect to server with timeout
                    headers = {
                        "Authorization": f"Bearer;{self.volcengine_app_access_token}",
                    }
                    websocket = await asyncio.wait_for(
                        websockets.connect(
                            self.endpoint,
                            additional_headers=headers,
                            max_size=10 * 1024 * 1024
                        ),
                        timeout=self._connection_timeout
                    )

                    # Use a fresh reqid per attempt to avoid duplicate request conflicts
                    request_payload = copy.deepcopy(base_request)
                    request_payload["request"]["reqid"] = request_id

                    # Send request
                    await full_client_request(websocket, json.dumps(request_payload).encode())

                    # Receive audio data with timeout
                    audio_data = await asyncio.wait_for(
                        self._receive_audio_data(websocket),
                        timeout=self._receive_timeout
                    )

                    # Success - return the audio data
                    return audio_data

                except Exception as e:
                    last_error = e
                    logger.warning(f"Audio generation attempt {attempt + 1}/{self._max_retries} failed (reqid: {request_id}): {str(e)}")

                    # Don't retry on the last attempt
                    if attempt < self._max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s, ...
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)

                finally:
                    # Always close websocket if it was created
                    if websocket is not None:
                        try:
                            await websocket.close()
                        except Exception:
                            pass  # Ignore close errors

            # All retries failed
            raise RuntimeError(
                f"Failed to generate audio after {self._max_retries} attempts. "
                f"Last error: {type(last_error).__name__}: {str(last_error)}"
            )

    @staticmethod
    def _get_audio_metadata(
        audio_data: bytes,
        audio_format: str,
        languages: List[str] = None,
    ) -> AudioFileMetaData:
        """
        Extract metadata from audio data.

        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format ('mp3' or 'wav')
            languages: List of language codes used in the audio

        Returns:
            AudioFileMetaData instance with extracted metadata
        """
        size_bytes = len(audio_data)
        mime_type = f"audio/{audio_format}"

        # Default values
        duration_seconds = None
        sample_rate_hz = None
        channels = None

        try:
            if audio_format == "wav":
                # Use built-in wave module for WAV files
                with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration_seconds = frames / float(rate) if rate > 0 else None
                    sample_rate_hz = rate
                    channels = wav_file.getnchannels()
            else:
                # Use av library for other formats (mp3, etc.)
                if av is None:
                    raise ImportError("av is required to extract non-WAV audio metadata")
                container = av.open(io.BytesIO(audio_data), format=audio_format)
                stream = container.streams.audio[0]

                if stream.duration and stream.time_base:
                    duration_seconds = float(stream.duration * stream.time_base)
                sample_rate_hz = stream.rate
                channels = stream.channels

                container.close()
        except Exception as e:
            logger.warning(f"Failed to extract audio metadata: {e}")

        return AudioFileMetaData(
            size_bytes=size_bytes,
            mime_type=mime_type,
            language=languages,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )

    def _load_volcengine_voices(self) -> List[Dict]:
        """Load volcengine voices from JSON file if not already loaded."""
        if self._volcengine_voices is not None:
            return self._volcengine_voices

        voices_file = os.path.join(
            os.path.dirname(__file__),
            "assets", "volcengine_voices.json"
        )

        if not os.path.exists(voices_file):
            raise FileNotFoundError(f"Volcengine voices file not found: {voices_file}")

        with open(voices_file, "r", encoding="utf-8") as f:
            self._volcengine_voices = json.load(f)

        return self._volcengine_voices

    def _filter_voice_candidates(
        self,
        languages: List[str] = None,
        gender: str = None
    ) -> List[Dict]:
        """
        Filter voice candidates based on languages and gender.

        Args:
            languages: List of language codes (e.g., ['zh-CN', 'en-US'])
            gender: 'male' or 'female'

        Returns:
            List of matching voice candidates
        """
        voices = self._load_volcengine_voices()
        candidates = []

        for voice in voices:
            # Filter by gender if specified
            if gender and voice.get("gender") != gender.lower():
                continue

            # Filter by language if specified
            if languages:
                voice_lang_codes = [lang["code"] for lang in voice.get("languages", [])]

                # Check if voice supports any of the requested languages
                if not any(lang in voice_lang_codes for lang in languages):
                    continue

                # For multilingual requests (e.g., Chinese + English mixed), prefer voices that support mixing
                if len(languages) > 1:
                    if not voice.get("support_mixed", False):
                        continue

            candidates.append(voice)

        return candidates

    def _format_candidates_for_prompt(self, candidates: List[Dict]) -> str:
        """Format voice candidates as a string for the LLM prompt."""
        formatted = []
        for i, voice in enumerate(candidates, 1):
            lang_info = ", ".join([f"{l['label']}({l['code']})" for l in voice.get("languages", [])])
            emotions_info = ""
            if voice.get("support_emotion") and voice.get("supported_emotions"):
                emotions_info = f"\n   Supported Emotions: {', '.join(voice['supported_emotions'])}"

            formatted.append(
                f"{i}. ID: {voice['id']}\n"
                f"   Name: {voice['name']}\n"
                f"   Gender: {voice['gender']}\n"
                f"   Languages: {lang_info}\n"
                f"   Supports Mixed Languages: {voice.get('support_mixed', False)}\n"
                f"   Supports Emotion: {voice.get('support_emotion', False)}{emotions_info}\n"
                f"   Hint: {voice.get('hint', 'N/A')}"
            )

        return "\n\n".join(formatted)

    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """
        Calculate similarity between two strings using Levenshtein distance ratio.
        Returns a value between 0 and 1, where 1 means identical.
        """
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Simple Levenshtein-based similarity
        len1, len2 = len(s1), len(s2)
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        # Create distance matrix
        distances = list(range(len1 + 1))
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
            distances = new_distances

        max_len = max(len1, len2)
        return 1 - (distances[-1] / max_len)

    def _find_best_matching_voice(self, llm_voice_id: str, candidates: List[Dict]) -> Optional[Dict]:
        """
        Find the best matching voice from candidates using string similarity.
        This handles cases where LLM might have minor typos in the ID.
        """
        if not candidates:
            return None

        best_match = None
        best_similarity = 0.0

        for candidate in candidates:
            similarity = self._string_similarity(llm_voice_id, candidate["id"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate

        # Only return match if similarity is reasonably high (> 0.7)
        if best_similarity > 0.7:
            return best_match

        return None

    async def _select_voice_with_llm(
        self,
        candidates: List[Dict],
        languages: List[str],
        gender: str,
        audio_hint: str
    ) -> Dict[str, Any]:
        """
        Use LLM to select the most appropriate voice from candidates.

        Returns:
            Dict with keys: voice_id, voice_name, emotion, reason
        """
        candidates_info = self._format_candidates_for_prompt(candidates)

        prompt = VOICE_SELECTION_PROMPT.format(
            languages=", ".join(languages) if languages else "Not specified",
            gender=gender or "Not specified",
            audio_hint=audio_hint or "Not specified",
            candidates_info=candidates_info
        )

        try:
            response = await self.llm.async_generate(prompt, max_tokens=1024)
            result = parse_json_from_llm_output(response.content)

            # Validate required fields
            if not result.get("voice_id"):
                raise ValueError("LLM response missing 'voice_id' field")

            return {
                "voice_id": result.get("voice_id", ""),
                "voice_name": result.get("voice_name"),
                "emotion": result.get("emotion"),
                "reason": result.get("reason")
            }
        except Exception as e:
            # Re-raise with more context, will be caught in __call__
            raise RuntimeError(f"LLM voice selection failed: {e}") from e

    def _get_default_voice_id(self, languages: List[str] = None, gender: str = None) -> str:
        """Get default voice ID based on language and gender preferences."""
        # Default to English if no language specified
        primary_lang = languages[0] if languages else "en-US"
        # Default to female if no gender specified
        selected_gender = gender or "female"

        # Try to get from config
        if primary_lang in DEFAULT_VOICE_CONFIG:
            lang_config = DEFAULT_VOICE_CONFIG[primary_lang]
            if selected_gender in lang_config:
                return lang_config[selected_gender]

        # Fallback to a safe default
        return "en_female_lauren_moon_bigtts"

    def _needs_custom_voice_selection(
        self,
        languages: List[str] = None,
        gender: str = None,
        audio_hint: str = None
    ) -> bool:
        """
        Determine if custom voice selection via LLM is needed.
        Returns False if all customization parameters are None.
        """
        return not (languages is None and gender is None and audio_hint is None)


PARAM_LANGUAGE_DESC = "Optional but recommended. The language of the audio file, by default is 'en-US', provide the language code to have a precise recognition result. Supported languages: 'en-US' (English US), 'zh-CN' (Chinese Mandarin), 'en-GB' (English UK), 'en-AU' (English Australia), 'ja-JP' (Japanese), 'es-ES' (Spanish)"

class AudioRecognitionTool(Tool):
    """
    Audio Recognition Tool (STT - Speech to Text) using Volcengine API.
    Converts audio files to text using Volcengine's speech recognition service.
    """
    name: str = "recognize_audio_to_text"
    description: str = (
        "Recognize and transcribe an audio file to text. "
        "Returns a dict containing 'success' (bool) and 'text' (the transcribed text content), and may also provide additional information such as 'utterances' (list of utterance segments with timestamps and speaker info). "
        "You should use this tool when dealing with tasks related to speech-to-text conversion, audio transcription, meeting minutes generation, or extracting text content from voice recordings."
    )
    extra_description: str = AUDIO_RECOGNITION_EXTRA_DESC.strip()
    inputs: Dict = {
        "audio_url": {
            "type": "string",
            "description": "Required. The URL of the audio file to recognize and transcribe.",
        },
        "audio_format": {
            "type": "string",
            "description": "Optional. The format of the audio file. The format of the audio file (e.g., wav, mp3).",
        },
        "language": {
            "type": "string",
            "description": PARAM_LANGUAGE_DESC,
        },
        "use_itn": {
            "type": "boolean",
            "description": "Optional. Whether to use inverse text normalization (ITN) to convert the recognized text to a more natural form (e.g., 'one hundred' -> '100'). Default is 'True'.",
        },
        "with_speaker_info": {
            "type": "boolean",
            "description": "Optional. Whether to include speaker diarization in the recognition result, identifying different speakers in the audio. Default is 'False'. Enable this when you need to distinguish between multiple speakers in the transcription.",
        }
    }
    required: Optional[List[str]] = ["audio_url"]

    def __init__(
        self,
        name: Optional[str] = None,
        volcengine_appid: Optional[str] = None,
        volcengine_app_access_token: Optional[str] = None,
        tool_user_id: Optional[str] = None,
        cluster: str = "volc_auc_common_flash",
        endpoint: str = "https://openspeech.bytedance.com/api/v1/auc",
        max_wait_time: int = 300,
        poll_interval: int = 2,
    ):
        """
        Initialize AudioRecognitionTool.

        Args:
            name: Custom name for the tool
            volcengine_appid: Volcengine application ID
            volcengine_app_access_token: Volcengine application access token
            tool_user_id: User ID for the tool, to distinguish different users in the same "appid" on volcengine platform
            cluster: STT cluster type, options include:
                - volc_auc_common_flash: Common audio (default)
                - volc_auc_meeting_flash: Meeting audio
                - volc_auc_tele_flash: Telephone audio
                - volc_auc_video_flash: Video audio
            endpoint: Volcengine STT API endpoint
            max_wait_time: Maximum wait time for task completion (seconds)
            poll_interval: Interval between status polling (seconds)
        """
        super().__init__()
        self.name = name or self.name
        self.volcengine_appid = volcengine_appid or os.getenv("VOLCENGINE_APPID")
        self.volcengine_app_access_token = volcengine_app_access_token or os.getenv("VOLCENGINE_APP_ACCESS_TOKEN")
        self.tool_user_id = tool_user_id or str(uuid.uuid4())
        self.cluster = cluster
        self.endpoint = endpoint
        self.max_wait_time = max_wait_time
        self.poll_interval = poll_interval

        self.check_external_provider_config()

        # STT task status codes mapping
        self.task_status_codes_mapping = {
            1000: "Success: recognition completed successfully.",

            1001: "Invalid request parameters: missing required fields, invalid field values, or duplicate request.",
            1002: "Access denied: invalid or expired token, or no permission to access the specified service.",
            1003: "Rate limit exceeded: current appid QPS exceeds the configured threshold.",
            1004: "Quota exceeded: current appid request count exceeds the allowed limit.",
            1005: "Server busy: service is overloaded and cannot process the request.",
            1006: "Request interrupted: the current request has expired or an error occurred.",

            1010: "Audio too long: audio duration exceeds the allowed threshold.",
            1011: "Audio too large: audio data size exceeds the allowed limit (single packet temporarily limited to 2 MB).",
            1012: "Invalid audio format: audio header is invalid or audio decoding failed.",
            1013: "Silent audio: no recognizable text detected in the audio.",
            1014: "Empty audio: downloaded audio content is empty.",
            1015: "Download failed: failed to download audio from the provided URL.",

            1020: "Recognition wait timeout: waiting for readiness timed out.",
            1021: "Recognition processing timeout: recognition processing exceeded the time limit.",
            1022: "Recognition error: an error occurred during the recognition process.",

            1099: "Unknown error: unclassified error.",

            2000: "Processing: the task is currently being processed.",
            2001: "Queued: the task is waiting in the queue."
        }
        self.task_status_pending_group = [2000, 2001]
        self.task_status_failed_group = [1001, 1002, 1003, 1004, 1005, 1006, 1010, 1011, 1012, 1013, 1014, 1015, 1020, 1021, 1022, 1099]
        self.task_status_success_group = [1000]

    def check_external_provider_config(self) -> None:
        """Validate required configuration parameters."""
        if self.volcengine_appid is None:
            raise ValueError("Volcengine appid is required")
        if self.volcengine_app_access_token is None:
            raise ValueError("Volcengine app access token is required")
        if self.endpoint is None:
            raise ValueError("Volcengine STT endpoint is required")

    def _get_cost(self, result) -> float:
        """
        Args:
            result(Dict[str, Any]): The raw result from Volcengine API, check https://www.volcengine.com/docs/6561/80820?lang=zh#_4-%E6%9F%A5%E8%AF%A2%E7%BB%93%E6%9E%9C
        cost in RMB
        """
        utterances = result.get("resp", {}).get("utterances", [])
        if not utterances:
            return 0
        last_utterance = utterances[-1]
        end_time = last_utterance.get("end_time", 0)
        cost = end_time / VOLCENGINE_STT_PRICE["post_paid"]["unit"] * VOLCENGINE_STT_PRICE["post_paid"]["price"]
        return cost

    async def __call__(
        self,
        audio_url: str,
        audio_format: str = "mp3",
        language: str = "en-US",
        use_itn: bool = True,
        with_speaker_info: bool = False,
    ) -> ToolResult:
        """
        Recognize audio from URL and return transcribed text.

        Args:
            audio_url: URL of the audio file to recognize
            audio_format: Format of the audio file (default: mp3)

        Returns:
            ToolResult containing the transcribed text or error message
        """
        tool_args = {
            "audio_url": audio_url,
            "audio_format": audio_format,
            "language": language,
            "use_itn": use_itn,
            "with_speaker_info": with_speaker_info,
        }
        metadata = ToolMetadata(tool_name=self.name, args=tool_args)

        try:
            result = await self.recognize_audio(**tool_args)
            # select contents to keep in the ToolResult instance
            if result['success']:
                result_to_keep = {
                    'success': result['success'],
                    'text': result['text'],
                }
            else:
                result_to_keep = {
                    'success': result['success'],
                    'error': result['error'],
                    'raw_result': result['raw_result'],
                }

            # get cost 
            cost = self._get_cost(result["raw_result"])
            metadata.add_cost_breakdown({"volcengine:stt__lu_yin_wen_jian_shi_bie": cost})

            return ToolResult(result=result_to_keep, metadata=metadata)
        except Exception as e:
            return ToolResult(
                result={"success": False, "error": f"Error recognizing audio: {e}"}, 
                metadata=metadata
            )

    async def recognize_audio(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Submit audio recognition task and wait for result.

        Args example:
            audio_url: URL of the audio file
            audio_format: Format of the audio file

        Returns:
            Dict containing recognition result or error, including:
            - success: True if task completed successfully, False otherwise
            - task_id: The ID of the task
            - text: The transcribed text
            - raw_result: The raw result from Volcengine API
        """
        # Submit the recognition task
        task_id = await self._submit_task(**kwargs)
        
        # Poll for result
        start_time = time.time()
        result = {}
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                # Check timeout first to prevent infinite loop when task stays pending
                elapsed_time = time.time() - start_time
                if elapsed_time > self.max_wait_time:
                    return {
                        "success": False,
                        "task_id": task_id,
                        "error": f"Task timeout after {self.max_wait_time} seconds",
                        "raw_result": result,
                    }

                await asyncio.sleep(self.poll_interval)

                # Query task with network error handling
                try:
                    result = await self._query_task(task_id, client)
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    logger.warning(f"Query task failed with network error: {e}, retrying...")
                    continue

                resp_code = result.get("resp", {}).get("code", -1)

                if resp_code in self.task_status_success_group:
                    # Task completed successfully
                    return {
                        "success": True,
                        "task_id": task_id,
                        "text": self._extract_text(result),
                        "raw_result": result,
                    }
                elif resp_code in self.task_status_failed_group:
                    # Task failed
                    error_msg = self.task_status_codes_mapping.get(resp_code, f"Unknown error code: {resp_code}")
                    return {
                        "success": False,
                        "task_id": task_id,
                        "error": f"Task failed with code: {resp_code}, error message: {error_msg}",
                        "raw_result": result,
                    }
                elif resp_code in self.task_status_pending_group:
                    # Task is still pending, continue polling
                    continue
                else:
                    # Unknown status code - log warning and continue polling
                    logger.warning(f"Unknown STT response code: {resp_code}, continuing to poll...")

    async def _submit_task(
        self,
        audio_url: str,
        audio_format: str,
        language: str,
        use_itn: bool,
        with_speaker_info: bool,
        max_retries: int = 3,
    ) -> str:
        """
        Submit audio recognition task to Volcengine.

        Args:
            audio_url: URL of the audio file
            audio_format: Format of the audio file
            max_retries: Maximum number of retry attempts for network errors

        Returns:
            Task ID for querying results
        """
        headers = {
            "Authorization": f"Bearer; {self.volcengine_app_access_token}",
        }

        request_payload = {
            "app": {
                "appid": self.volcengine_appid,
                "token": self.volcengine_app_access_token,
                "cluster": self.cluster,
            },
            "user": {
                "uid": self.tool_user_id,
            },
            "audio": {
                "format": audio_format,
                "url": audio_url,
            },
            "additions": {
                "with_speaker_info": with_speaker_info,
                "language": language,
                "use_itn": use_itn,
            },
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.endpoint}/submit",
                        content=json.dumps(request_payload),
                        headers=headers,
                    )
                    response.raise_for_status()
                    resp_dict = response.json()

                if "resp" not in resp_dict or "id" not in resp_dict.get("resp", {}):
                    raise RuntimeError(f"Failed to submit task: {resp_dict}")

                return resp_dict["resp"]["id"]
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
                logger.warning(f"Submit task failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

        raise RuntimeError(f"Failed to submit task after {max_retries} attempts: {last_error}")

    async def _query_task(self, task_id: str, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
        """
        Query the status and result of a recognition task.

        Args:
            task_id: The task ID returned from submit
            client: Optional httpx client (if provided, will reuse it; otherwise creates a new one)

        Returns:
            Dict containing task status and result
        """
        headers = {
            "Authorization": f"Bearer; {self.volcengine_app_access_token}",
        }

        query_payload = {
            "appid": self.volcengine_appid,
            "token": self.volcengine_app_access_token,
            "id": task_id,
            "cluster": self.cluster,
        }

        if client is not None:
            # Reuse provided client
            response = await client.post(
                f"{self.endpoint}/query",
                content=json.dumps(query_payload),
                headers=headers,
            )
        else:
            # Create new client
            async with httpx.AsyncClient(timeout=30.0) as new_client:
                response = await new_client.post(
                    f"{self.endpoint}/query",
                    content=json.dumps(query_payload),
                    headers=headers,
                )
        
        response.raise_for_status()
        return response.json()

    def _extract_text(self, result: Dict[str, Any]) -> str:
        """
        Extract transcribed text from recognition result.

        Args:
            result: Raw result from Volcengine API

        Returns:
            Transcribed text string
        """
        try:
            # The result structure may vary, try common paths
            resp = result.get("resp", {})

            # Try to get text from utterances
            utterances = resp.get("utterances", [])
            if utterances:
                texts = [u.get("text", "") for u in utterances]
                return "".join(texts)

            # Try direct text field
            if "text" in resp:
                return resp["text"]

            # Try result field
            if "result" in resp:
                return resp["result"]

            # Return empty string if no text found
            return ""
        except Exception:
            return ""

class AudioMergeTool(Tool):
    """
    Audio Merge Tool for concatenating multiple audio files into a single audio file.
    """
    name: str = "merge_audio_files"
    description: str = (
        "Merge multiple audio files into a single audio file by concatenating them in the specified order. "
        "Accepts a list of audio URLs and returns the URL of the merged audio file. "
        "Supports common audio formats including mp3 and wav. "
        "Returns a dict containing 'success' (bool), 'audio_url' (URL link to the merged audio file), "
        "and 'audio_metadata' (technical details: duration, sample_rate, channels, mime_type). "
        "You should use this tool when you need to combine multiple audio segments into one continuous audio, "
        "such as merging dialogue lines from different speakers, combining podcast segments, or creating audio compilations."
    )
    inputs: Dict = {
        "audio_urls": {
            "type": "array",
            "description": "Required. A list of audio file URLs to merge. The audio files will be concatenated in the order they appear in the list.",
            "items": {
                "type": "string"
            }
        },
        "audio_output": {
            "type": "object",
            "description": "Optional. Output configuration for the merged audio file.",
            "properties": {
                "audio_name": {
                    "type": "string",
                    "description": "The name of the output audio file (without extension). Must use ASCII characters only. If not provided, a default name will be generated."
                },
                "audio_format": {
                    "type": "string",
                    "description": "The output audio format. Supported formats: 'mp3', 'wav'. Default: 'mp3'.",
                    "enum": ["mp3", "wav"]
                }
            }
        }
    }
    required: Optional[List[str]] = ["audio_urls"]

    def __init__(
        self,
        name: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        save_path: str = "./audio_files/volcengine",
        max_concurrent_downloads: int = 5,
        max_file_size_mb: float = 20.0,
        max_total_size_mb: float = 100.0,
        max_audio_count: int = 50,
        download_timeout: int = 240,
        max_download_retries: int = 3,
    ):
        super().__init__()
        self.name = name or self.name
        self.max_concurrent_downloads = max_concurrent_downloads
        if self.max_concurrent_downloads < 1:
            logger.warning(f"`max_concurrent_downloads`({max_concurrent_downloads}) is less than 1. Fixing it to 1.")
            self.max_concurrent_downloads = 1

        # Resource limits
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.max_total_size_bytes = int(max_total_size_mb * 1024 * 1024)
        self.max_audio_count = max_audio_count
        self.download_timeout = download_timeout
        self.max_download_retries = max_download_retries

        # storage handler for storing audio files
        if storage_handler is None:
            if os.getenv("SUPABASE_URL_STORAGE", None):
                storage_handler = SupabaseStorageHandler()
            else:
                storage_handler = LocalStorageHandler(base_path=save_path)

        self.storage_handler = storage_handler

    async def __call__(
        self,
        audio_urls: list,
        audio_output: dict = None,
        **kwargs
    ) -> ToolResult:
        tool_args = {
            "audio_urls": audio_urls,
            "audio_output": audio_output
        }
        metadata = ToolMetadata(tool_name=self.name, args=tool_args)

        # Validate input
        if not audio_urls or len(audio_urls) < 2:
            return ToolResult(
                result={
                    "success": False,
                    "error": "At least 2 audio URLs are required for merging."
                },
                metadata=metadata
            )

        # Validate count limit
        if len(audio_urls) > self.max_audio_count:
            return ToolResult(
                result={
                    "success": False,
                    "error": (
                        f"Too many audio files: {len(audio_urls)} exceeds limit of {self.max_audio_count}. "
                        f"To merge all files, split them into batches of {self.max_audio_count} or fewer, "
                        f"merge each batch separately using multiple merge_audio_files calls, "
                        f"then merge the resulting files together in a final merge operation."
                    )
                },
                metadata=metadata
            )

        # Parse audio output options
        default_audio_name = f"merged_audio_{uuid.uuid4().hex[:8]}"
        if audio_output:
            raw_audio_name = audio_output.get("audio_name")
            if raw_audio_name is None:
                audio_name = default_audio_name
            elif not isinstance(raw_audio_name, str) or not raw_audio_name.strip():
                return ToolResult(
                    result={
                        "success": False,
                        "error": "audio_name cannot be empty. Please provide a valid audio_name in audio_output parameter, or omit it to use auto-generated name.",
                    },
                    metadata=metadata
                )
            else:
                audio_name = raw_audio_name
            audio_format = audio_output.get("audio_format") or "mp3"
        else:
            audio_name = default_audio_name
            audio_format = "mp3"

        if audio_format not in ["mp3", "wav"]:
            return ToolResult(
                result={
                    "success": False,
                    "error": f"Unsupported audio_format '{audio_format}'. Supported formats are: ['mp3', 'wav']. Please update audio_output parameter with a valid audio_format value."
                },
                metadata=metadata
            )

        # Create a shared HTTP client for all downloads (enables connection pooling)
        async with httpx.AsyncClient(timeout=self.download_timeout) as http_client:
            # Validate file sizes before downloading
            try:
                await self._validate_audio_sizes(audio_urls, http_client)
            except ValueError as e:
                return ToolResult(
                    result={
                        "success": False,
                        "error": str(e)
                    },
                    metadata=metadata
                )

            # Download all audio files with retry logic and concurrency control
            logger.info(f"Downloading {len(audio_urls)} audio files...")
            try:
                audio_segments = await self._download_audio_files(audio_urls, http_client)
            except Exception as e:
                return ToolResult(
                    result={
                        "success": False,
                        "error": f"Error downloading audio files: {str(e)}"
                    },
                    metadata=metadata
                )

        # Merge audio files in thread pool to avoid blocking event loop
        logger.info(f"Merging {len(audio_segments)} audio segments...")
        try:
            # Offload CPU-bound merge to thread pool
            merged_audio = await asyncio.to_thread(
                self._merge_audio_segments,
                audio_segments,
                audio_format
            )
        except Exception as e:
            return ToolResult(
                result={
                    "success": False,
                    "error": f"Error merging audio files: {str(e)}"
                },
                metadata=metadata
            )

        # Save merged audio to storage
        full_audio_name = f"{audio_name}.{audio_format}"
        try:
            save_result = self.storage_handler.save(full_audio_name, merged_audio)
            # Check if save operation failed (when success key is explicitly present)
            if "success" in save_result and not save_result["success"]:
                error_msg = save_result.get("error", "Unknown error during save")
                return ToolResult(
                    result={
                        "success": False,
                        "error": f"Error saving merged audio: {error_msg}"
                    },
                    metadata=metadata
                )
        except Exception as e:
            return ToolResult(
                result={
                    "success": False,
                    "error": f"Error saving merged audio: {str(e)}"
                },
                metadata=metadata
            )

        # Get audio metadata (reuse AudioGenerationTool's static method)
        audio_metadata = AudioGenerationTool._get_audio_metadata(merged_audio, audio_format)

        # Build result
        result = {
            "success": True,
            "audio_url": save_result.get("url") or save_result.get("file_path"),
            "audio_metadata": audio_metadata.to_dict(),
            "merged_count": len(audio_urls)
        }

        # Add warning if name was changed
        parsed_url = urlparse(result["audio_url"])
        file_name = os.path.basename(parsed_url.path)
        basename = os.path.splitext(file_name)[0]
        raw_audio_name = audio_output.get("audio_name") if audio_output else None
        if raw_audio_name and basename != raw_audio_name:
            result["warning"] = f"The saved audio name is different from the requested: requested '{raw_audio_name}', actually saved as '{basename}', use the url in the 'audio_url' field to access the actually saved audio!"

        return ToolResult(result=result, metadata=metadata)

    def _merge_audio_segments(self, audio_segments: List[bytes], output_format: str) -> bytes:
        """
        Merge multiple audio segments into a single audio file.
        This is a CPU-bound operation that should be run in a thread pool.

        Args:
            audio_segments: List of audio data as bytes
            output_format: Output audio format ('mp3' or 'wav')

        Returns:
            Merged audio data as bytes
        """
        if not audio_segments:
            raise ValueError("No audio segments to merge")

        if len(audio_segments) == 1:
            return audio_segments[0]

        # Use pydub for audio merging (more reliable for concatenation)
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pydub is required for audio merging. Install it with: pip install pydub")

        # Load and merge segments with progress tracking
        merged = None
        total_segments = len(audio_segments)

        for i, audio_data in enumerate(audio_segments):
            try:
                logger.info(f"Processing audio segment {i + 1}/{total_segments}")

                # Create AudioSegment from bytes
                segment = AudioSegment.from_file(io.BytesIO(audio_data))

                if merged is None:
                    merged = segment
                else:
                    # Concatenate audio segments
                    merged = merged + segment

            except Exception as e:
                raise RuntimeError(f"Failed to process audio segment {i}: {str(e)}")

        # Free up memory from audio segments after processing
        audio_segments.clear()

        if merged is None:
            raise ValueError("Failed to merge audio segments")

        # Export merged audio to bytes
        logger.info("Exporting merged audio...")
        output_buffer = io.BytesIO()
        merged.export(output_buffer, format=output_format)
        output_buffer.seek(0)

        result = output_buffer.read()
        logger.info(f"Merge complete. Output size: {len(result) / 1024:.2f} KB")

        return result

    async def _validate_audio_sizes(self, audio_urls: List[str], client: httpx.AsyncClient) -> None:
        """
        Validate audio file sizes before downloading using HEAD requests.

        Args:
            audio_urls: List of audio URLs to validate
            client: Shared httpx.AsyncClient for connection pooling

        Raises:
            ValueError: If size limits are exceeded
        """
        total_size = 0
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)

        async def check_size(url: str, index: int) -> int:
            """Get file size using HEAD request."""
            async with semaphore:
                try:
                    response = await client.head(url, follow_redirects=True)
                    response.raise_for_status()

                    size_header = response.headers.get("content-length")
                    if not size_header:
                        logger.warning(f"HEAD response missing content-length for URL {index}, skipping pre-check")
                        return 0

                    try:
                        size = int(size_header)
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid content-length for URL {index}, skipping pre-check: {size_header}")
                        return 0

                    if size > self.max_file_size_bytes:
                        raise ValueError(
                            f"Audio file at index {index} exceeds size limit: "
                            f"{size / 1024 / 1024:.2f}MB > {self.max_file_size_bytes / 1024 / 1024:.2f}MB"
                        )

                    return size
                except ValueError:
                    raise
                except Exception as e:
                    # If HEAD fails, skip size check (will validate during download)
                    logger.warning(f"Could not check size for URL {index}, skipping validation: {str(e)}")
                    return 0

        # Check all file sizes in parallel
        tasks = [check_size(url, i) for i, url in enumerate(audio_urls)]
        sizes = await asyncio.gather(*tasks, return_exceptions=True)

        for i, size_result in enumerate(sizes):
            if isinstance(size_result, Exception):
                # Re-raise ValueError (size limit exceeded)
                if isinstance(size_result, ValueError):
                    raise size_result
                # Log other exceptions but continue
                logger.warning(f"Error checking size for audio {i}: {size_result}")
            else:
                total_size += size_result

        # Check total size limit
        if total_size > self.max_total_size_bytes:
            raise ValueError(
                f"Total audio size exceeds limit: "
                f"{total_size / 1024 / 1024:.2f}MB > {self.max_total_size_bytes / 1024 / 1024:.2f}MB"
            )

    async def _download_audio_files(self, audio_urls: List[str], client: httpx.AsyncClient) -> List[bytes]:
        """
        Download audio files with retry logic and concurrency control.

        Args:
            audio_urls: List of audio URLs to download
            client: Shared httpx.AsyncClient for connection pooling

        Returns:
            List of audio data as bytes, ordered by input URL list

        Raises:
            Exception: If download fails after retries
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)

        async def download_with_retry(index: int, url: str) -> tuple:
            """Download a single audio file with retry logic."""
            async with semaphore:
                last_error = None

                for attempt in range(self.max_download_retries):
                    try:
                        logger.info(f"Downloading audio {index + 1}/{len(audio_urls)} (attempt {attempt + 1})")
                        response = await client.get(url)
                        response.raise_for_status()

                        audio_data = response.content

                        # Validate downloaded size
                        if len(audio_data) == 0:
                            raise ValueError(f"Downloaded audio at index {index} is empty")

                        if len(audio_data) > self.max_file_size_bytes:
                            raise ValueError(
                                f"Downloaded audio at index {index} exceeds size limit: "
                                f"{len(audio_data) / 1024 / 1024:.2f}MB > "
                                f"{self.max_file_size_bytes / 1024 / 1024:.2f}MB"
                            )

                        logger.info(f"Successfully downloaded audio {index + 1}/{len(audio_urls)} ({len(audio_data) / 1024:.2f} KB)")
                        return index, audio_data

                    except (httpx.HTTPError, httpx.TimeoutException, ValueError) as e:
                        last_error = e
                        logger.warning(f"Download attempt {attempt + 1}/{self.max_download_retries} failed for audio {index}: {str(e)}")

                        # Don't retry on validation errors
                        if isinstance(e, ValueError):
                            raise

                        # Exponential backoff before retry
                        if attempt < self.max_download_retries - 1:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)

                # All retries failed
                raise RuntimeError(
                    f"Failed to download audio at index {index} after {self.max_download_retries} attempts: "
                    f"{str(last_error)}"
                )

        # Download all files in parallel with concurrency control
        tasks = [download_with_retry(i, url) for i, url in enumerate(audio_urls)]
        download_results = await asyncio.gather(*tasks)

        # Sort by index to maintain order
        download_results.sort(key=lambda x: x[0])
        return [data for _, data in download_results]


class AudioToolkit(Toolkit):
    """
    A toolkit that bundles audio-related tools (TTS, STT, and audio merging).
    """

    def __init__(
        self,
        name: str = "AudioToolkit",
        storage_handler: FileStorageHandler = None,
        **kwargs
    ):
        # Obtain Volcengine credentials
        openrouter_key = kwargs.get("openrouter_key", None) or os.getenv("OPENROUTER_API_KEY")
        volcengine_appid = kwargs.get("volcengine_appid", None) or os.getenv("VOLCENGINE_APPID")
        volcengine_app_access_token = kwargs.get("volcengine_app_access_token", None) or os.getenv("VOLCENGINE_APP_ACCESS_TOKEN")
        save_path = kwargs.get("save_path", None) or "."

        # Initialize storage handler if not provided
        if storage_handler is None:
            storage_handler = LocalStorageHandler(save_path)

        # Create tools
        audio_generation_tool = AudioGenerationTool(
            openrouter_key=openrouter_key,
            volcengine_appid=volcengine_appid,
            volcengine_app_access_token=volcengine_app_access_token,
            storage_handler=storage_handler,
        )
        audio_recognition_tool = AudioRecognitionTool(
            volcengine_appid=volcengine_appid,
            volcengine_app_access_token=volcengine_app_access_token,
        )
        audio_merge_tool = AudioMergeTool(
            storage_handler=storage_handler,
            max_concurrent_downloads=kwargs.get("max_concurrent_downloads", 5),
            max_file_size_mb=kwargs.get("max_file_size_mb", 20.0),
            max_total_size_mb=kwargs.get("max_total_size_mb", 100.0),
            max_audio_count=kwargs.get("max_audio_count", 50),
            download_timeout=kwargs.get("download_timeout", 240),
            max_download_retries=kwargs.get("max_download_retries", 3),
        )

        tools = [audio_generation_tool, audio_recognition_tool, audio_merge_tool]

        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        self.storage_handler = storage_handler
