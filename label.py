import base64
import os
from openai import OpenAIError  # Import specific error for better handling
from llm import LabelLLM
from pathlib import Path
import json


class LabelGenerator(LabelLLM):
    """
    Extends the LabelLLM client with utility methods for handling image files.
    It can encode local images to base64 and generate descriptions directly
    from a file path.
    """

    @staticmethod
    def encode_image(image_path: Path) -> str:
        """
        Encodes an image file to a base64 string.
        This is a static method as it doesn't depend on any instance state.

        Args:
            image_path: The path to the image file to encode.

        Returns:
            A base64-encoded string of the image.

        Raises:
            FileNotFoundError: If the image_path does not exist.
        """
        with image_path.open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def parse_json_str(json_string: str) -> dict | None:
        """
        Parses a JSON string and returns a dictionary. Returns None on failure.
        Made static as it doesn't depend on instance state.
        """
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON. Reason: {e}")
            print(f"Invalid string was: {json_string}")
            return None

    def generate_label_from_image(self, image_path: Path) -> str:
        """
        Encodes an image from a path and gets its description.

        Args:
            image_path: The path to the image file.

        Returns:
            The LLM-generated description of the image.
        """
        base64_image = self.encode_image(image_path)
        return self.run(base64_image)

    def generate_json_from_image(self, image_path: Path) -> dict | None:
        """
        ** workflow:
        1. Encodes an image from a path.
        2. Gets the raw string response from the LLM.
        3. Parses the string response into a dictionary.

        Args:
            image_path: The path to the image file.

        Returns:
            A dictionary parsed from the LLM's response, or None if parsing fails.
        """
        # Get the raw string response from the model
        raw_response = self.generate_label_from_image(image_path)

        # Parse the response and return the result
        return self.parse_json_str(raw_response)


if __name__ == "__main__":
    try:
        context_prompt_file = Path("prompts") / "context_prompt.txt"
        # Instantiate the generator.
        label_generator = LabelGenerator(
            llm_deployment_name="o4-mini-labeling",
            context_prompt_file=context_prompt_file,
        )

        # Define the path to the target image.
        image_path = Path("frames") / "frame_0.jpg"

        print(f"Generating label for image: {image_path}...")

        # get the response from llm model
        response = label_generator.generate_json_from_image(image_path)

        print("\n--- Generated Description ---")
        print(f"{type(response)}: {response}")
        print("---------------------------\n")

    except FileNotFoundError:
        print(f"Error: The image file was not found at '{image_path}'.")
    except OpenAIError as e:
        # Handle potential API errors from the 'openai' library.
        print(f"An error occurred with the OpenAI API: {e}")
    except Exception as e:
        # Catch any other unexpected errors.
        print(f"An unexpected error occurred: {e}")
