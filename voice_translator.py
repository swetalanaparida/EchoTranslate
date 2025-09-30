# Import required libraries
import os
import numpy as np
import gradio as gr
import assemblyai as aai
from translate import Translator
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pathlib import Path
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY")
ELEVENLABS_KEY = os.getenv("ELEVENLABS_KEY")
VOICE_ID = os.getenv("VOICE_ID")


# Main pipeline: voice → text → translations → voices
def voice_to_voice(audio_file):
    transcript = transcribe_audio(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        raise gr.Error(transcript.error)
    else:
        transcript = transcript.text

    list_translations = translate_text(transcript)
    generated_audio_paths = []

    for translation in list_translations:
        translated_audio_file_name = text_to_speech(translation)
        path = Path(translated_audio_file_name)
        generated_audio_paths.append(path)

    return (
        generated_audio_paths[0], generated_audio_paths[1], generated_audio_paths[2],
        generated_audio_paths[3], generated_audio_paths[4], generated_audio_paths[5],
        list_translations[0], list_translations[1], list_translations[2],
        list_translations[3], list_translations[4], list_translations[5]
    )


# Audio → text
def transcribe_audio(audio_file):
    aai.settings.api_key = ASSEMBLYAI_KEY
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)
    return transcript


# English → multiple translations
def translate_text(text: str) -> str:
    languages = ["ru", "tr", "sv", "de", "es", "ja"]
    list_translations = []

    for lan in languages:
        translator = Translator(from_lang="en", to_lang=lan)
        translation = translator.translate(text)
        list_translations.append(translation)

    return list_translations


# Text → speech
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


# Gradio interface setup
input_audio = gr.Audio(
    sources=["microphone"],
    type="filepath",
    show_download_button=True,
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
)

with gr.Blocks() as demo:
    gr.Markdown("## Record yourself in English and immediately receive voice translations.")

    # Input section
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                show_download_button=True,
                waveform_options=gr.WaveformOptions(
                    waveform_color="#01C6FF",
                    waveform_progress_color="#0066B4",
                    skip_length=2,
                    show_controls=False,
                ),
            )
            with gr.Row():
                submit = gr.Button("Submit", variant="primary")
                btn = gr.ClearButton(audio_input, "Clear")

    # Output section (6 audios + 6 texts)
    with gr.Row():
        with gr.Group():
            tr_output = gr.Audio(label="Turkish", interactive=False)
            tr_text = gr.Markdown()
        with gr.Group():
            sv_output = gr.Audio(label="Swedish", interactive=False)
            sv_text = gr.Markdown()
        with gr.Group():
            ru_output = gr.Audio(label="Russian", interactive=False)
            ru_text = gr.Markdown()

    with gr.Row():
        with gr.Group():
            de_output = gr.Audio(label="German", interactive=False)
            de_text = gr.Markdown()
        with gr.Group():
            es_output = gr.Audio(label="Spanish", interactive=False)
            es_text = gr.Markdown()
        with gr.Group():
            jp_output = gr.Audio(label="Japanese", interactive=False)
            jp_text = gr.Markdown()

    # Link button with function
    output_components = [
        ru_output, tr_output, sv_output, de_output, es_output, jp_output,
        ru_text, tr_text, sv_text, de_text, es_text, jp_text
    ]
    submit.click(fn=voice_to_voice, inputs=audio_input, outputs=output_components, show_progress=True)
        

if __name__ == "__main__":
    demo.launch()
