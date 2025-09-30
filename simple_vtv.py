# Import required libraries
import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
from pathlib import Path
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()  
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY")
ELEVENLABS_KEY = os.getenv("ELEVENLABS_KEY")
VOICE_ID = os.getenv("VOICE_ID")


# Main pipeline: voice → text → translation → voice
def voice_to_voice(audio_file):
    transcription_response = audio_transcription(audio_file)

    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)
    else:
        text = transcription_response.text

    es_translation, tr_translation, ja_translation = text_translation(text)

    es_audi_path = text_to_speech(es_translation)
    tr_audi_path = text_to_speech(tr_translation)
    ja_audi_path = text_to_speech(ja_translation)

    return Path(es_audi_path), Path(tr_audi_path), Path(ja_audi_path)


# Audio → text (AssemblyAI)
def audio_transcription(audio_file):
    aai.settings.api_key = ASSEMBLYAI_KEY
    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)
    return transcription


# English text → Spanish, Turkish, Japanese
def text_translation(text):
    es_text = Translator(from_lang="en", to_lang="es").translate(text)
    tr_text = Translator(from_lang="en", to_lang="tr").translate(text)
    ja_text = Translator(from_lang="en", to_lang="ja").translate(text)
    return es_text, tr_text, ja_text


# Text → speech (ElevenLabs)
def text_to_speech(text: str) -> str:
    client = ElevenLabs(api_key=ELEVENLABS_KEY)
    save_file_path = f"{uuid.uuid4()}.mp3"
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.text_to_speech.convert(
                voice_id=VOICE_ID,
                optimize_streaming_latency="0",
                output_format="mp3_22050_32",
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.8,
                    style=0.5,
                    use_speaker_boost=True,
                ),
            )
            with open(save_file_path, "wb") as f:
                for chunk in response:
                    if chunk:
                        f.write(chunk)
            return save_file_path

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise e


# Gradio interface
audio_input = gr.Audio(sources=["microphone"], type="filepath")

demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[
        gr.Audio(label="Spanish"),
        gr.Audio(label="Turkish"),
        gr.Audio(label="Japanese")
    ]
)

if __name__ == "__main__":
    demo.launch()
