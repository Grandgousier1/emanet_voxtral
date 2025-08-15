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
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoProcessor,
    VoxtralForConditionalGeneration,
)
from utils.gpu_utils import free_cuda_mem, get_gpu_info
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class SimpleSegment:
    """A simple container for a transcription segment."""
    start: float
    end: float
    text: str


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

    # Pipeline options
    parser.add_argument(
        "--pipeline-type",
        type=str,
        default=config.get("pipeline_options", {}).get("type", "voxtral_single_step"),
        choices=["two_step", "voxtral_single_step"],
        help="The type of pipeline to run.",
    )

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

    # Model and device options
    parser.add_argument(
        "--voxtral-model-path",
        type=str,
        default=config.get("model_paths", {}).get(
            "voxtral", "mistralai/Voxtral-Small-24B-2507"
        ),
        help="Path to the Voxtral model (used in both pipelines).",
    )
    parser.add_argument(
        "--faster-whisper-model-path",
        type=str,
        default=config.get("model_paths", {}).get(
            "faster_whisper", "ctranslate2-4you/whisper-large-v2-ct2-int8"
        ),
        help="Path to the faster-whisper ASR model (fallback for 'two_step' pipeline).",
    )
    parser.add_argument(
        "--nllb-model-path",
        type=str,
        default=config.get("model_paths", {}).get(
            "nllb_translation", "facebook/nllb-200-distilled-600M"
        ),
        help="Path to the NLLB translation model (for 'two_step' pipeline).",
    )
    parser.add_argument(
        "--transcription-device",
        type=str,
        default=config.get("transcription_options", {}).get("device", "cuda"),
        help="Device to use for transcription models (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--translation-device",
        type=str,
        default=config.get("translation_options", {}).get("device", "cuda"),
        help="Device to use for translation models (e.g., 'cuda', 'cpu').",
    )

    # Language and translation options
    parser.add_argument(
        "--translation-model-type",
        type=str,
        default=config.get("translation_options", {}).get("model_type", "nllb"),
        choices=["nllb"],
        help="Type of translation model for the 'two_step' pipeline (currently only 'nllb' is supported).",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default=config.get("translation_options", {}).get("source_lang", "tur_Latn"),
        help="Source language for 'two_step' NLLB translation.",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default=config.get("translation_options", {}).get("target_lang", "French"),
        help="Target language. Use full names for Voxtral (e.g., 'French'), codes for NLLB (e.g., 'fra_Latn').",
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
        logging.error("This could be due to a network issue, a private/deleted video, or YouTube blocking the request.")
        logging.error("TIP: For YouTube, try using a cookies file with --cookies to bypass potential restrictions.")

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
        segments, info = whisper_model.transcribe(audio_path, beam_size=5)

        # faster-whisper returns a generator, convert segments to a list
        segment_list = list(segments)

        transcription = "".join([segment.text for segment in segment_list])
        logging.info("faster-whisper transcription successful.")

        # Create SimpleSegment objects for SRT generation
        simple_segments = [SimpleSegment(start=s.start, end=s.end, text=s.text) for s in segment_list]

        return {"text": transcription, "segments": simple_segments}
    except Exception as e:
        logging.error(f"faster-whisper transcription failed: {e}")
        return None
    finally:
        if "whisper_model" in locals():
            del whisper_model
        free_cuda_mem()


def transcribe_and_translate_with_voxtral(model, processor, audio_path, target_lang, device):
    """
    Performs transcription and translation in a single step using Voxtral's
    audio instruct capabilities.
    """
    logging.info(f"Attempting single-step transcription and translation to '{target_lang}' with Voxtral.")
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": f"Translate the following audio to {target_lang}."},
                ],
            }
        ]

        inputs = processor.apply_chat_template(conversation, return_tensors="pt")
        inputs = inputs.to(device, dtype=torch.bfloat16)

        outputs = model.generate(**inputs, max_new_tokens=1024)
        decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        translated_text = decoded_outputs[0]
        logging.info("Voxtral single-step translation successful.")

        # Voxtral does not provide timed segments, so we create one segment for the whole audio.
        # We need the audio duration to create a valid SRT.
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate

        single_segment = SimpleSegment(start=0, end=duration, text=translated_text)

        return {"text": translated_text, "segments": [single_segment]}

    except Exception as e:
        logging.error(f"Voxtral single-step translation failed: {e}")
        return None
    finally:
        # No need to del model/processor as they are managed outside
        free_cuda_mem()


def translate_text(text, tokenizer, model, model_type, src_lang, tgt_lang, device):
    """
    Translates text using a pre-loaded NLLB model and tokenizer.
    This function is only used in the 'two_step' pipeline.
    """
    logging.info(f"Translating text to {tgt_lang} using {model_type} model.")
    if model_type != 'nllb':
        logging.error(f"The 'two_step' pipeline currently only supports 'nllb' for translation, but got '{model_type}'.")
        return None

    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=1024,
        )
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        logging.info("Translation successful.")
        return translation
    except Exception as e:
        logging.error(f"NLLB Translation failed: {e}")
        return None


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

    # --- Model Loading ---
    voxtral_model = None
    voxtral_processor = None
    translation_model = None
    translation_tokenizer = None

    if args.pipeline_type == "voxtral_single_step":
        logging.info(f"Loading Voxtral model for single-step translation: {args.voxtral_model_path}")
        try:
            voxtral_processor = AutoProcessor.from_pretrained(args.voxtral_model_path)
            voxtral_model = VoxtralForConditionalGeneration.from_pretrained(
                args.voxtral_model_path, torch_dtype=torch.bfloat16, device_map=args.transcription_device
            )
            logging.info("Voxtral model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Voxtral model: {e}")
            logging.error("Cannot proceed with single-step pipeline. Please check model path and dependencies.")
            return # Exit if the main model can't be loaded

    elif args.pipeline_type == "two_step":
        if args.target_lang and args.source_lang != args.target_lang:
            logging.info(f"Loading translation model for two-step pipeline: {args.nllb_model_path}")
            try:
                if args.translation_model_type == 'nllb':
                    translation_tokenizer = AutoTokenizer.from_pretrained(args.nllb_model_path, src_lang=args.source_lang)
                    translation_model = AutoModelForSeq2SeqLM.from_pretrained(args.nllb_model_path).to(args.translation_device)
                # The mistral option for two-step is removed for simplicity.
                # It can be re-added if necessary.
                logging.info("Translation model loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load translation model: {e}")
                logging.warning("Translation will be skipped.")

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

    for i, url in enumerate(video_urls):
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing video {i+1}/{len(video_urls)}: {url}")
        logging.info(f"{'='*60}")
        start_time = time.time()

        # Step 1: Download
        logging.info("\n[Step 1/4] Downloading Media...")
        video_path, audio_path = download_media(
            url, args.output_dir, args.download_video, args.download_audio, args.cookies
        )

        if not audio_path or not os.path.exists(audio_path):
            logging.error(f"Audio download failed for {url}. Skipping to next video.")
            continue
        logging.info("[Step 1/4] Download complete.")

        base_filename = os.path.splitext(os.path.basename(audio_path))[0]

        if args.pipeline_type == "voxtral_single_step":
            # --- Single-Step Pipeline ---
            logging.info("\n[Step 2/2] Performing Single-Step Transcription and Translation...")
            result = transcribe_and_translate_with_voxtral(
                voxtral_model, voxtral_processor, audio_path, args.target_lang, args.transcription_device
            )

            if not result or "segments" not in result:
                logging.error(f"Single-step translation failed for {url}. Skipping.")
                os.remove(audio_path)
                continue

            translated_srt_path = os.path.join(
                args.output_dir, f"{base_filename}_{args.target_lang}.srt"
            )
            generate_srt(result["segments"], translated_srt_path)
            logging.info(f"Translated SRT file saved to: {translated_srt_path}")

        elif args.pipeline_type == "two_step":
            # --- Two-Step Pipeline ---
            # Step 2: Transcription
            logging.info("\n[Step 2/4] Transcribing Audio...")
            transcription = None
            # 1. Try Voxtral model for transcription
            logging.info("Attempting transcription with Voxtral...")
            transcription = transcribe_with_voxtral(
                args.voxtral_model_path, audio_path, args.transcription_device
            )
            # 2. Fallback to faster-whisper
            if not transcription:
                logging.warning("Voxtral transcription failed, falling back to faster-whisper...")
                transcription = transcribe_with_faster_whisper(
                    args.faster_whisper_model_path, audio_path, args.transcription_device
                )

            if not transcription or "segments" not in transcription:
                logging.error(f"All transcription attempts failed for {url}. Skipping.")
                os.remove(audio_path)
                continue
            logging.info("[Step 2/4] Transcription complete.")

            # Step 3: Generate original SRT
            logging.info("\n[Step 3/4] Generating Original Language SRT File...")
            srt_path = os.path.join(args.output_dir, f"{base_filename}.srt")
            generate_srt(transcription["segments"], srt_path)
            logging.info(f"Original SRT file saved to: {srt_path}")

            # Step 4: Translation
            if translation_model and translation_tokenizer:
                logging.info("\n[Step 4/4] Translating Segments...")
                translated_segments = []

                for segment in tqdm(transcription["segments"], desc="Translating segments"):
                    translated_text = translate_text(
                        text=segment.text,
                        tokenizer=translation_tokenizer,
                        model=translation_model,
                        model_type=args.translation_model_type,
                        src_lang=args.source_lang,
                        tgt_lang=args.target_lang,
                        device=args.translation_device,
                    )
                    if translated_text:
                        translated_segments.append(
                            SimpleSegment(start=segment.start, end=segment.end, text=translated_text)
                        )
                    else:
                        logging.warning(f"Translation failed for segment: '{segment.text.strip()}'.")

                if translated_segments:
                    translated_srt_path = os.path.join(
                        args.output_dir, f"{base_filename}_{args.target_lang}.srt"
                    )
                    generate_srt(translated_segments, translated_srt_path)
                    logging.info(f"Translated SRT file saved to: {translated_srt_path}")
                else:
                    logging.warning("Translation resulted in no segments. No translated SRT file generated.")

        logging.info("\n[Cleanup] Removing temporary files...")
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logging.info(f"-> Deleted audio file: {audio_path}")
        if 'video_path' in locals() and video_path and os.path.exists(video_path):
            os.remove(video_path)
            logging.info(f"-> Deleted video file: {video_path}")
        free_cuda_mem()

        end_time = time.time()
        logging.info(f"\nFinished processing in {end_time - start_time:.2f} seconds.")

    logging.info("Batch processing complete.")
    logging.info("\n" + "="*60)
    logging.info("Done. All videos processed.")
    logging.info("NOTE: To free up more disk space, you can manually clear the Hugging Face model cache, usually located at `~/.cache/huggingface/hub`.")
    logging.info("The 'make clean' command can also be used to clear the 'output' directory.")
    logging.info("="*60 + "\n")


if __name__ == "__main__":
    main()
