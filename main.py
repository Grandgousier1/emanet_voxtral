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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MistralForCausalLM
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
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
        "--translation-model-type",
        type=str,
        default=config.get("translation_options", {}).get("model_type", "nllb"),
        choices=["nllb", "mistral"],
        help="Type of translation model to use ('nllb' or 'mistral').",
    )
    parser.add_argument(
        "--translation-model-path",
        type=str,
        default=config.get("model_paths", {}).get(
            "translation", "facebook/nllb-200-distilled-600M"
        ),
        help="Path or Hugging Face identifier for the translation model.",
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


def translate_text(text, tokenizer, model, model_type, src_lang, tgt_lang, device):
    """Translates text using a pre-loaded model and tokenizer."""
    logging.info(f"Translating text to {tgt_lang} using {model_type} model.")
    try:
        if model_type == 'nllb':
            inputs = tokenizer(text, return_tensors="pt").to(device)
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                max_length=1024,
            )
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        elif model_type == 'mistral':
            # This is the high-quality prompt for translating Turkish drama dialogues to French
            # This prompt is designed to be general while capturing the user's specific request for quality.
            lang_map = {"fra_Latn": "French", "eng_Latn": "English"} # Simple mapping for now
            target_language_name = lang_map.get(tgt_lang, tgt_lang)
            system_prompt = (
                f"You are an expert translator. Your task is to translate the following text into {target_language_name}. "
                "The text is dialogue from a dramatic series. It is crucial to preserve the emotional tone, "
                "cultural nuances, and dramatic weight of the original dialogue. "
                "The translation must be fluid, natural, and faithful to the original's intent and subtext."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            # This logic is based on the model card for mistralai/Mistral-Small-3.2-24B-Instruct-2506
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
            input_ids = torch.tensor([tokenized.tokens]).to(device)

            translated_tokens = model.generate(
                input_ids=input_ids,
                max_new_tokens=1024,
            )[0]
            # Exclude the prompt from the output
            translation = tokenizer.decode(translated_tokens[len(tokenized.tokens):])

        logging.info("Translation successful.")
        return translation
    except Exception as e:
        logging.error(f"Translation failed: {e}")
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

    # Load translation model and tokenizer once
    translation_model = None
    translation_tokenizer = None
    if args.target_lang and args.source_lang != args.target_lang:
        logging.info(f"Loading translation model: {args.translation_model_path}")
        try:
            if args.translation_model_type == 'nllb':
                translation_tokenizer = AutoTokenizer.from_pretrained(args.translation_model_path, src_lang=args.source_lang)
                translation_model = AutoModelForSeq2SeqLM.from_pretrained(args.translation_model_path).to(args.translation_device)
            elif args.translation_model_type == 'mistral':
                # Note: This part cannot be tested in the current environment due to space constraints.
                translation_tokenizer = MistralTokenizer.from_hf_hub(args.translation_model_path)
                translation_model = MistralForCausalLM.from_pretrained(
                    args.translation_model_path, torch_dtype=torch.bfloat16
                ).to(args.translation_device)
            logging.info("Translation model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load translation model: {e}")
            logging.error("This could be due to an incorrect model path, insufficient VRAM, or missing dependencies.")
            logging.warning("Transcription will proceed, but translation will be skipped.")

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

        # Step 2: Transcription
        logging.info("\n[Step 2/4] Transcribing Audio...")
        transcription = None
        # 1. Try Voxtral Small
        if not transcription:
            logging.info("Attempting transcription with Voxtral Small...")
            transcription = transcribe_with_voxtral(
                args.voxtral_small_model_path, audio_path, args.transcription_device
            )
        # 2. Try Voxtral Mini
        if not transcription:
            logging.warning("Voxtral Small failed, trying Voxtral Mini...")
            transcription = transcribe_with_voxtral(
                args.voxtral_mini_model_path, audio_path, args.transcription_device
            )
        # 3. Try faster-whisper
        if not transcription:
            logging.warning("Voxtral Mini failed, trying faster-whisper...")
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
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
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
