from dotenv import load_dotenv
from openai import AzureOpenAI
from dataclasses import dataclass
import os
from pathlib import Path

load_dotenv()


@dataclass(frozen=True)
class AzureOpenAIConfig:
    """
    An immutable dataclass to store configuration for the Azure OpenAI service.
    It reads values from environment variables, and its instances cannot be
    modified after creation.
    """

    # Define fields with type hints and default values read from environment variables.
    # The dataclass decorator will automatically generate the __init__ method from these.
    azure_endpoint: str = os.getenv("ENDPOINT")
    api_key: str = os.getenv("API_KEY")
    api_version: str = os.getenv("API_VERSION")

    def __post_init__(self):
        """
        Perform validation after the instance has been initialized.
        This method is automatically called by the dataclass after __init__.
        """
        if not all((self.azure_endpoint, self.api_key, self.api_version)):
            raise ValueError("Missing environment variables!!!")


class LabelLLM:
    """A client for interacting with an Azure OpenAI vision model to describe images."""

    def __init__(
        self,
        context_prompt_file: Path,
        llm_deployment_name: str = "o4-mini-labeling",
    ):
        """
        Initializes the LabelLLM client.

        Args:
            llm_deployment_name: The name of the Azure OpenAI model deployment to use.
            context_prompt_file: The path to the file containing the context prompt.
        """

        # Load Azure credentials and endpoint settings.
        self.config = AzureOpenAIConfig()
        self.llm_model = llm_deployment_name

        # Cache the prompt content
        try:
            self.context_prompt = context_prompt_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Error: Context prompt file not found at '{context_prompt_file}'")
            raise  # Re-raise the exception to stop initialization if the file is critical.

        # Create an instance of the AzureOpenAI client to communicate with the service.
        self.client = AzureOpenAI(
            api_key=self.config.api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.azure_endpoint,
        )

    def run(self, base64_image: str) -> str:
        """
        Sends a base64-encoded image to the vision model and returns a text description.

        Args:
            base64_image: A string containing the base64-encoded image data.

        Returns:
            A string containing the model's generated description of the image.
        """
        # Call the chat completions API with a multi-modal payload.
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        # The text part of the prompt.
                        {
                            "type": "text",
                            "text": self.context_prompt,
                        },
                        # The image part of the prompt, formatted as a data URI.
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=500,  # Optional: Set a limit for the response length.
        )

        # Extract and return the text content from the first choice in the response.
        return response.choices[0].message.content
