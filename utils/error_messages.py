"""
Centralized error messages for the application to enforce DRY principle.
"""
import sys
from typing import Optional

# Using a dictionary to store error messages and their solutions.
# The key is a unique identifier for the error.
# The value is a tuple containing the formatted error message and the solution.

ERROR_MESSAGES = {
    "SECURITY_PATH_NOT_ALLOWED": (
        "{file_type} path not allowed for security reasons: {path}",
        "Use relative paths within the current project directory only."
    ),
    "FILE_NOT_FOUND": (
        "{file_type} not found: {path}",
        "Check the file path and ensure you have the necessary permissions."
    ),
    "BATCH_FILE_EMPTY": (
        "No valid URLs or paths found in the batch file.",
        "Please add at least one URL or file path to the batch file."
    ),
    "NO_INPUT_SPECIFIED": (
        "No input specified.",
        "Use --url for a single video/file or --batch-list for batch processing. Use --help for more options."
    ),
    "MODEL_LOAD_FAILED": (
        "Voxtral model could not be loaded.",
        "Check your model installation, configuration, and available GPU memory."
    ),
    "PROCESSING_FAILED": (
        "Processing failed due to errors.",
        "Please check the errors logged above for more details."
    ),
    "NETWORK_ERROR": (
        "A network error occurred.",
        "Check your internet connection and the validity of the URL."
    ),
    "EXTERNAL_COMMAND_FAILED": (
        "An external command failed to execute.",
        "Check the command, its parameters, and any related logs."
    ),
    "PREFLIGHT_CHECKS_FAILED": (
        "Preflight checks failed - execution aborted.",
        Fix errors above or use --force to override (not recommended)
    ),
    "GENERIC_PROCESSING_ERROR": (
        "Error processing segment {segment_index}",
        "No solution available."
    ),
    "YT_DLP_INFO_EXTRACT_FAILED": (
        "Failed to extract video information for URL: {url}",
        "The video may be private, deleted, or there might be network issues."
    ),
}

def get_error_message(key: str, **kwargs) -> (str, str):
    """
    Retrieves and formats an error message and its solution from the central dictionary.

    Args:
        key: The identifier for the error message (e.g., "FILE_NOT_FOUND").
        **kwargs: Placeholders to format into the message string (e.g., file_type="Cookie file", path="/path/to/file").

    Returns:
        A tuple containing the formatted (message, solution).
        Returns a generic error if the key is not found.
    """
    message_template, solution_template = ERROR_MESSAGES.get(
        key,
        ("An unknown error occurred: {key}", "Please report this issue.")
    )
    
    try:
        formatted_message = message_template.format(**kwargs)
        formatted_solution = solution_template.format(**kwargs)
    except KeyError as e:
        # This helps debug if a placeholder is missing in the call
        formatted_message = f"Error formatting message '{key}': Missing placeholder {e}"
        formatted_solution = "Please check the code calling get_error_message."

    return formatted_message, formatted_solution


class ErrorReporter:
    """
    A class to report errors using a CLIFeedback instance.
    This decouples the error message definitions from the feedback reporting mechanism.
    """
    def __init__(self, feedback):
        self.feedback = feedback

    def report(self, key: str, details: Optional[str] = None, **kwargs):
        """
        Reports an error using the configured feedback instance.

        Args:
            key: The identifier for the error message.
            details: Optional additional details for the error.
            **kwargs: Placeholders for the error message.
        """
        message, solution = get_error_message(key, **kwargs)
        self.feedback.error(message, solution=solution, details=details)

    def report_and_exit(self, key: str, details: Optional[str] = None, **kwargs):
        """
        Reports a critical error and exits the application.

        Args:
            key: The identifier for the error message.
            details: Optional additional details for the error.
            **kwargs: Placeholders for the error message.
        """
        message, solution = get_error_message(key, **kwargs)
        self.feedback.critical(message, solution=solution, details=details)
        sys.exit(1)