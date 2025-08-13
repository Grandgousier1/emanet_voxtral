import argparse
import datetime
import gc
import logging
import os
import subprocess
import sys
import time
import torch
import torchaudio
import yt_dlp
import yaml
from utils.gpu_utils import free_cuda_mem, get_gpu_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        return None


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated Subtitling Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config = load_config()
    if not config:
        # Fallback to defaults if config fails to load
        config = {
            "model_paths": {},
            "transcription_options": {},
            "translation_options": {},
            "youtube_dl_options": {},
        }

    # Batch processing
    parser.add_argument(
        "--batch-list", type=str, help="Path to a text file with a list of video URLs."
    )

    # YouTube download options
    parser.add_argument(
        "--download-video",
        action="store_true",
        default=config.get("youtube_dl_options", {}).get("download_video", True),
        help="Download the video file.",
    )
    parser.add_argument(
        "--download-audio",
        action="store_true",
        default=config.get("youtube_dl_options", {}).get("download_audio", True),
        help="Download the audio file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.get("youtube_dl_options", {}).get("output_dir", "output"),
        help="Directory to save downloaded files.",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=config.get("youtube_dl_options", {}).get("cookies", None),
        help="Path to a cookies.txt file for yt-dlp.",
    )

    # Transcription options
    parser.add_argument(
        "--voxtral-small-model-path",
        type=str,
        default=config.get("model_paths", {}).get(
            "voxtral_small", "collabora/whisperspeech_asr_small_en"
        ),
        help="Path to the Voxtral Small ASR model.",
    )
    parser.add_argument(
        "--voxtral-mini-model-path",
        type=str,
        default=config.get("model_paths", {}).get(
            "voxtral_mini", "collabora/whisperspeech_asr_mini_en"
        ),
        help="Path to the Voxtral Mini ASR model.",
    )
    parser.add_argument(
        "--faster-whisper-model-path",
        type=str,
        default=config.get("model_paths", {}).get(
            "faster_whisper", "ctranslate2-4you/whisper-small-en-ct2-int8"
        ),
        help="Path to the faster-whisper ASR model.",
    )
    parser.add_argument(
        "--transcription-device",
        type=str,
        default=config.get("transcription_options", {}).get("device", "cuda"),
        help="Device to use for transcription (e.g., 'cuda', 'cpu').",
    )

    # Translation options
    parser.add_argument(
        "--nllb-model-path",
        type=str,
        default=config.get("model_paths", {}).get(
            "nllb", "facebook/nllb-200-distilled-600M"
        ),
        help="Path to the NLLB translation model.",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default=config.get("translation_options", {}).get("source_lang", "eng_Latn"),
        help="Source language for translation.",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default=config.get("translation_options", {}).get("target_lang", "fra_Latn"),
        help="Target language for translation.",
    )
    parser.add_argument(
        "--translation-device",
        type=str,
        default=config.get("translation_options", {}).get("device", "cuda"),
        help="Device to use for translation (e.g., 'cuda', 'cpu').",
    )

    return parser.parse_args()


def download_media(url, output_dir, download_video, download_audio, cookies_file):
    """Downloads video and/or audio using yt-dlp."""
    logging.info(f"Starting download for URL: {url}")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "retries": 10,
        "fragment_retries": 10,
        "ignoreerrors": True,
        "postprocessors": [],
    }

    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
        logging.info(f"Using cookies from {cookies_file}")

    if download_audio:
        ydl_opts["postprocessors"].append(
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        )
        if not download_video:
            ydl_opts["format"] = "bestaudio/best"

    video_path, audio_path = None, None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            if info_dict:
                base_path = ydl.prepare_filename(info_dict).rsplit(".", 1)[0]
                if download_video:
                    video_path = f"{base_path}.mp4"
                    if not os.path.exists(video_path):
                        # Sometimes the extension is different
                        for ext in [".webm", ".mkv", ".flv"]:
                            if os.path.exists(f"{base_path}{ext}"):
                                video_path = f"{base_path}{ext}"
                                break
                    logging.info(f"Video downloaded to: {video_path}")
                if download_audio:
                    audio_path = f"{base_path}.wav"
                    logging.info(f"Audio extracted to: {audio_path}")
            else:
                logging.error(f"Failed to download or extract info for URL: {url}")

    except Exception as e:
        logging.error(f"An error occurred during download: {e}")

    return video_path, audio_path


def transcribe_with_voxtral(model, audio_path, device):
    """Transcribes audio using a Voxtral model."""
    from whisperspeech.asr import ASREngine

    logging.info(f"Attempting transcription with Voxtral model: {model}")
    try:
        asr = ASREngine(model, device=device)
        transcription = asr.transcribe(audio_path)
        logging.info("Voxtral transcription successful.")
        return transcription
    except Exception as e:
        logging.error(f"Voxtral transcription failed: {e}")
        return None
    finally:
        if "asr" in locals():
            del asr
        free_cuda_mem()


def transcribe_with_faster_whisper(model, audio_path, device):
    """Transcribes audio using faster-whisper."""
    from faster_whisper import WhisperModel

    logging.info(f"Attempting transcription with faster-whisper model: {model}")
    try:
        # Determine compute_type based on device
        compute_type = "int8"
        if device == "cuda":
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
                compute_type = (
                    "float16"  # Use float16 for better performance on modern GPUs
                )
            else:
                compute_type = "int8"

        whisper_model = WhisperModel(model, device=device, compute_type=compute_type)
        segments, _ = whisper_model.transcribe(audio_path, beam_size=5)
        transcription = "".join([segment.text for segment in segments])
        logging.info("faster-whisper transcription successful.")
        return {"text": transcription, "segments": list(segments)}
    except Exception as e:
        logging.error(f"faster-whisper transcription failed: {e}")
        return None
    finally:
        if "whisper_model" in locals():
            del whisper_model
        free_cuda_mem()


def translate_text(text, model, src_lang, tgt_lang, device):
    """Translates text using NLLB."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logging.info(f"Translating text from {src_lang} to {tgt_lang}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, src_lang=src_lang)
        translator = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        translated_tokens = translator.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=1024,
        )
        translation = tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )[0]
        logging.info("Translation successful.")
        return translation
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return None
    finally:
        if "translator" in locals():
            del translator
        if "tokenizer" in locals():
            del tokenizer
        free_cuda_mem()


def format_timestamp(seconds):
    """Formats seconds into SRT timestamp format."""
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def generate_srt(segments, output_path):
    """Generates an SRT file from transcription segments."""
    logging.info(f"Generating SRT file at: {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                start_time = format_timestamp(segment.start)
                end_time = format_timestamp(segment.end)
                f.write(f"{i + 1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment.text.strip()}\n\n")
        logging.info("SRT file generation successful.")
    except Exception as e:
        logging.error(f"Failed to generate SRT file: {e}")


def main():
    """Main function to run the subtitling pipeline."""
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    model_cache = {}

    def get_model(model_name, model_loader, *loader_args):
        if model_name not in model_cache:
            logging.info(f"Loading {model_name}...")
            model_cache[model_name] = model_loader(*loader_args)
            logging.info(f"{model_name} loaded.")
        return model_cache[model_name]

    if args.batch_list:
        try:
            with open(args.batch_list, "r") as f:
                video_urls = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logging.error(f"Batch file not found: {args.batch_list}")
            return
    else:
        logging.error("No batch list provided. Please specify with --batch-list.")
        return

    for url in video_urls:
        logging.info(f"Processing video: {url}")
        start_time = time.time()

        _, audio_path = download_media(
            url, args.output_dir, args.download_video, args.download_audio, args.cookies
        )

        if not audio_path or not os.path.exists(audio_path):
            logging.error(f"Audio download failed for {url}. Skipping.")
            continue

        # ASR Fallback Chain
        transcription = None
        # 1. Try Voxtral Small
        if not transcription:
            transcription = transcribe_with_voxtral(
                args.voxtral_small_model_path, audio_path, args.transcription_device
            )
        # 2. Try Voxtral Mini
        if not transcription:
            transcription = transcribe_with_voxtral(
                args.voxtral_mini_model_path, audio_path, args.transcription_device
            )
        # 3. Try faster-whisper
        if not transcription:
            transcription = transcribe_with_faster_whisper(
                args.faster_whisper_model_path, audio_path, args.transcription_device
            )

        if not transcription or "segments" not in transcription:
            logging.error(f"All transcription attempts failed for {url}. Skipping.")
            os.remove(audio_path)  # Clean up audio file
            continue

        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        srt_path = os.path.join(args.output_dir, f"{base_filename}.srt")
        generate_srt(transcription["segments"], srt_path)

        # Translation
        if args.target_lang and args.source_lang != args.target_lang:
            translated_text = translate_text(
                transcription["text"],
                args.nllb_model_path,
                args.source_lang,
                args.target_lang,
                args.translation_device,
            )
            if translated_text:
                translated_srt_path = os.path.join(
                    args.output_dir, f"{base_filename}_{args.target_lang}.srt"
                )
                # This is a simplification; a real implementation would need
                # to align translated sentences with original timestamps.
                # For now, we'll just save the full translated text.
                try:
                    with open(
                        translated_srt_path.replace(".srt", ".txt"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(translated_text)
                    logging.info(
                        f"Full translated text saved to {translated_srt_path.replace('.srt', '.txt')}"
                    )
                except Exception as e:
                    logging.error(f"Could not save translated text: {e}")

        # Cleanup
        os.remove(audio_path)
        free_cuda_mem()

        end_time = time.time()
        logging.info(
            f"Finished processing {url} in {end_time - start_time:.2f} seconds."
        )

    logging.info("Batch processing complete.")


if __name__ == "__main__":
    main()
