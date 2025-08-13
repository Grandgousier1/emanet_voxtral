import torch
import gc
import logging
import subprocess


def free_cuda_mem():
    """
    Frees up GPU memory by clearing PyTorch's cache and running garbage collection.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Cleared PyTorch CUDA cache.")
    except Exception as e:
        logging.error(f"Error clearing PyTorch CUDA cache: {e}")

    try:
        gc.collect()
        logging.info("Completed garbage collection.")
    except Exception as e:
        logging.error(f"Error during garbage collection: {e}")


def get_gpu_info():
    """
    Retrieves and logs information about the available GPUs using nvidia-smi.
    """
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. GPU info cannot be retrieved.")
        return

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_info = result.stdout.strip().split("\n")
        logging.info("--- GPU Information ---")
        for i, line in enumerate(gpu_info):
            name, total_mem, free_mem = line.split(", ")
            logging.info(f"GPU {i}:")
            logging.info(f"  Name: {name}")
            logging.info(f"  Total Memory: {total_mem} MiB")
            logging.info(f"  Free Memory: {free_mem} MiB")
        logging.info("-----------------------")
    except FileNotFoundError:
        logging.warning(
            "`nvidia-smi` command not found. Cannot display GPU information."
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing nvidia-smi: {e.stderr}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while getting GPU info: {e}")
