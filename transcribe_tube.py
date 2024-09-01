import gradio as gr
import yt_dlp
import os
import torchaudio
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
import pysrt
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import torch
import re
from typing import Any, Mapping, Optional, Text
import yaml
import shutil

# Load configuration from YAML file
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Extract configuration values
huggingface_token = config['huggingface']['token']
transcribe_model_path = config['huggingface']['transcribe_model_path']
diarization_model = config['huggingface']['diarization_model']
language_choices = config['languages']

# define result folder
result_folder = "result"
os.makedirs(result_folder, exist_ok=True)

# Custom hook class to show progress using tqdm
class TqdmProgressHook:
    """Hook to show progress of each internal step using tqdm"""

    def __init__(self):
        self.current_step_bar = None  # Placeholder for the current tqdm progress bar

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.current_step_bar:
            self.current_step_bar.close()

    def __call__(
        self,
        step_name: Text,
        step_artifact: Any,
        file: Optional[Mapping] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        if completed is None:
            completed = total = 1
        if not hasattr(self, "step_name") or step_name != self.step_name:
            self.step_name = step_name
            if self.current_step_bar:
                self.current_step_bar.close()
            self.current_step_bar = tqdm(total=total, desc=f"{step_name}")
        if self.current_step_bar:
            self.current_step_bar.update(1)

def format_selector(ctx):
    """Select the best video and the best audio that won't result in an mkv.
    Ensures the output format is mp4."""
    formats = ctx.get('formats')[::-1]
    best_video = next(
        f for f in formats if f['vcodec'] != 'none' and f['acodec'] == 'none')
    audio_ext = 'm4a'
    best_audio = next(f for f in formats if f['acodec'] !=
                      'none' and f['vcodec'] == 'none' and f['ext'] == audio_ext)

    yield {
        'format_id': f'{best_video["format_id"]}+{best_audio["format_id"]}',
        'ext': 'mp4',
        'requested_formats': [best_video, best_audio],
        'protocol': f'{best_video["protocol"]}+{best_audio["protocol"]}'
    }


# Function to convert seconds to pysrt.SubRipTime
def convert_seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return pysrt.SubRipTime(hours=hours, minutes=minutes, seconds=secs, milliseconds=millis)

# Helper function to extract audio from a video file
def extract_audio_from_video(video_path, output_wav_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_wav_path, codec='pcm_s16le')

# Function to simulate the transcription task with progress
def transcribe_video(input_video, language, progress=gr.Progress(track_tqdm=True)):
    os.makedirs("audio_chunks", exist_ok=True)

    # copy input video file
    # Extract the filename and extension from the input video path
    filename, ext = os.path.splitext(os.path.basename(input_video))
    # If the filename doesn't have an extension, add .mp4
    if not ext:
        ext = '.mp4'  # Add .mp4 as the default extension
    # Construct the full path for the output video
    video = os.path.join(result_folder, filename + ext)
    shutil.copy(input_video, video)
    
    wav_path = os.path.join(result_folder,"tmp.wav")
    srt_output_path = os.path.join(result_folder, f"{filename}.srt")

    extract_audio_from_video(video, wav_path)

    pipeline = Pipeline.from_pretrained(diarization_model, use_auth_token=huggingface_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    whisper_pipe = hf_pipeline("automatic-speech-recognition", model=transcribe_model_path,
                               device=0 if torch.cuda.is_available() else -1, generate_kwargs={"language": language, "task": "transcribe"})

    waveform, sample_rate = torchaudio.load(wav_path)

    with TqdmProgressHook() as hook:
        diarization = pipeline(wav_path, hook=hook)

    subs = pysrt.SubRipFile()
    total_segments = len(list(diarization.itertracks(yield_label=True)))
    progress_bar = tqdm(total=total_segments, desc="Transcribing")

    chunk_index = 0
    chunk_files = []
    sentence_splitter = re.compile(r'(?<=[.!?]) +')

    for speech_turn, _, _ in diarization.itertracks(yield_label=True):
        seg_start = speech_turn.start
        seg_end = speech_turn.end
        chunk_waveform = waveform[:, int(seg_start * sample_rate):int(seg_end * sample_rate)]
        chunk_filename = f"audio_chunks/temp_chunk_{chunk_index}.wav"
        torchaudio.save(chunk_filename, chunk_waveform, sample_rate)
        result = whisper_pipe(chunk_filename)
        transcription_text = result['text'].strip()
        sentences = sentence_splitter.split(transcription_text)

        total_duration_ms = (seg_end - seg_start) * 1000
        total_chars = sum(len(sentence) for sentence in sentences)
        current_start_time_ms = seg_start * 1000

        for sentence in sentences:
            words = sentence.split()
            if len(words) > 8:
                sentence_chunks = [words[i:i + 8] for i in range(0, len(words), 8)]
                total_words_in_sentence = len(words)
                sentence_ratio = len(sentence) / total_chars
                sentence_duration_ms = total_duration_ms * sentence_ratio

                for chunk in sentence_chunks:
                    chunk_word_count = len(chunk)
                    chunk_duration_ms = sentence_duration_ms * (chunk_word_count / total_words_in_sentence)
                    end_time_chunk_ms = current_start_time_ms + chunk_duration_ms
                    start_srt_time = convert_seconds_to_srt_time(current_start_time_ms / 1000)
                    end_srt_time = convert_seconds_to_srt_time(end_time_chunk_ms / 1000)

                    subtitle = pysrt.SubRipItem(index=len(subs) + 1, start=start_srt_time, end=end_srt_time, text=' '.join(chunk).strip())
                    subs.append(subtitle)
                    current_start_time_ms = end_time_chunk_ms
            else:
                sentence_ratio = len(sentence) / total_chars
                sentence_duration_ms = total_duration_ms * sentence_ratio
                end_time_sentence_ms = current_start_time_ms + sentence_duration_ms
                start_srt_time = convert_seconds_to_srt_time(current_start_time_ms / 1000)
                end_srt_time = convert_seconds_to_srt_time(end_time_sentence_ms / 1000)

                subtitle = pysrt.SubRipItem(index=len(subs) + 1, start=start_srt_time, end=end_srt_time, text=sentence.strip())
                subs.append(subtitle)
                current_start_time_ms = end_time_sentence_ms

        chunk_index += 1
        chunk_files.append(chunk_filename)
        progress_bar.update(1)

    progress_bar.close()
    subs.save(srt_output_path, encoding='utf-8')

    for chunk_file in chunk_files:
        os.remove(chunk_file)
    os.remove(wav_path)

    return (video, srt_output_path), (video, srt_output_path), gr.update(value=video, visible=True), gr.update(value=srt_output_path, visible=True)

# Function to clear the video player and remove the SRT file
def clear_output(video_and_srt_paths):

    video, srt_output_path = video_and_srt_paths

    if os.path.exists(srt_output_path):
        os.remove(srt_output_path)

    # only remove from video file in result folder, not from orignal local video file
    if os.path.exists(video):
        os.remove(video)

    return gr.update(value=None), gr.update(value=None), gr.update(visible=False), gr.update(visible=False),\
         gr.update(value=None, label="Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=example"), gr.update(value=None, visible=False),\
         gr.update(value=None)

def download_and_update(url):

    # define download youtube video file name
    output_file_from_download = os.path.join(result_folder, "download_file.mp4")

    if os.path.exists(output_file_from_download):
        os.remove(output_file_from_download)
    ydl_opts = {
        'format': format_selector,
        "outtmpl": output_file_from_download,
        "merge_output_format": "mp4",
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_file_from_download

def prepare_to_download():
    return "Downloading and Processing Videos...", gr.update(visible=True)

def complete_status():
    return "Download completed."

def show_uploaded_video(video_file):
    return video_file.name

# Define the Gradio interface
def create_interface():
    """Creates and launches the Gradio interface for the application."""
    with gr.Blocks() as iface:
        gr.Markdown("# TranscribeTube")

        video_and_srt_paths = gr.State()

        with gr.Row():
            with gr.Column(scale=1):
                youtube_url_input = gr.Textbox(
                    label="Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=example")
                download_button = gr.Button("Download Video")
                status_text = gr.Textbox(
                    label="Download Status", interactive=False, visible=False)
                video_input = gr.Video(label="Upload a video")
                language_dropdown = gr.Dropdown(
                    label="Select Transcription Language",
                    choices=language_choices,
                    value="English"
                )
                transcribe_button = gr.Button("Transcribe")
                clear_button = gr.Button("Clear")

            with gr.Column(scale=2):
                video_output = gr.Video(label="Video Output", interactive=False)
                download_video_button = gr.File(
                    label="Download Video", visible=False)
                download_sub_button = gr.File(
                    label="Download Subtitles", visible=False)

        # Link functions to Gradio buttons
        transcribe_button.click(
            fn=transcribe_video,
            inputs=[video_input, language_dropdown],
            outputs=[video_and_srt_paths, video_output, download_video_button, download_sub_button]
        )

        clear_button.click(
            fn=clear_output,
            inputs=[video_and_srt_paths],
            outputs=[video_input, video_output, download_sub_button, download_video_button,
                     youtube_url_input, status_text, video_and_srt_paths]
        )

        download_button.click(
            fn=prepare_to_download,
            inputs=None,
            outputs=[status_text, status_text]
        ).then(
            fn=download_and_update,
            inputs=[youtube_url_input],
            outputs=[video_input]
        ).then(
            fn=complete_status,
            inputs=None,
            outputs=[status_text]
        )

    iface.launch(share=False)

# Main entry point of the script
if __name__ == "__main__":
    create_interface()