import base64
from openai import OpenAIError  # Import specific error for better handling
from llm import LabelLLM
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd


class LabelGenerator(LabelLLM):
    """
    Extends the LabelLLM client with utility methods for handling image files.
    It can encode local images to base64 and generate descriptions directly
    from a file path.
    """

    def __init__(
        self,
        context_prompt_file: Path,
        input_dir: Path,
        output_dir: Path,
        llm_deployment_name: str = "o4-mini-labeling",
    ):
        super().__init__(context_prompt_file, llm_deployment_name)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self._validate_path()

    def _validate_path(self):
        if not (isinstance(self.input_dir, Path) and isinstance(self.output_dir, Path)):
            raise ValueError("input_dir and output_dir must be Path objects")

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

        Note:
            This method will raise exceptions (e.g., FileNotFoundError, OpenAIError) if any step fails.
        """

        # Get the raw string response from the model
        # If errors like FileNotFoundError or API errors occur here,
        # the exception will be propagated up to the caller.
        raw_response = self.generate_label_from_image(image_path)

        # Parse the response and return the result
        # If parsing fails, this will return None.
        return self.parse_json_str(raw_response)

    def generate_json_from_multiple_images(
        self, max_workers: int = 10
    ) -> OrderedDict[str, dict | None]:
        """Generates JSON labels for multiple images concurrently using a thread pool.

        This method scans an input directory for image files, submits each image
        to a ThreadPoolExecutor for processing, and collects the resulting
        JSON data in the order the images were submitted.

        Args:
            max_workers (int): The maximum number of worker threads to use for
                concurrent processing. Defaults to 10.

        Returns:
            OrderedDict[str, dict | None]: An ordered dictionary mapping image
                filenames to their corresponding JSON data (as a dict).
            None : if no images are found in the input directory.
        """
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
        # self.input_dir.glob("*.jpg") returns a generator.
        images = list(self.input_dir.glob("*.*"))
        images = [image for image in images if image.suffix.lower() in image_extensions]
        if not images:
            print(f"No images found in {self.input_dir}")
            return

        results = OrderedDict()
        # multi-threading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # futures: Creates a dictionary mapping Future objects (keys) to their corresponding image Path objects (values).
            # executor.submit(self.generate_json_from_image, image) : 작업을 예약하고, Future 라는 작업 예약증(작업 결과를 가져올 수 있는 객체) 를 반환
            # executor.submit(...): Schedules the function to be executed and returns a Future object, which represents the pending result.
            futures = {
                executor.submit(self.generate_json_from_image, image): image
                for image in images
            }
            # as_completed(futures) : 작업이 완료된 순서대로 Future 객체를 반환
            # as_completed(futures): Yields Future objects from the given set as they complete.
            for future in tqdm(
                as_completed(futures), total=len(images), desc="Processing ..."
            ):
                # 예약증(future)으로 원본 이미지의 (Path 객체)를 찾습니다.
                # Retrieve the original image Path object using the completed Future object.
                image = futures[future]
                try:
                    # future.result(): Retrieves the actual result of the task, i.e., the return value from self.generate_json_from_image.
                    response = future.result()
                    # Stores the result in the results dictionary with the image name as the key.
                    results[image.name] = response
                except FileNotFoundError:
                    print(
                        f"❌ ERROR: File not found for '{image.name}'. Skipping this file."
                    )
                # OpenAIError: Authentication, rate limit, server error, etc.
                except OpenAIError as e:
                    print(f"❌ ERROR: OpenAI API error for '{image.name}': {e}")
                # llm response is not valid json.
                except json.JSONDecodeError as e:
                    print(
                        f"❌ ERROR: Failed to parse JSON response for '{image.name}'."
                    )
                # Other unexpected errors.
                except Exception as e:
                    print(f"❌ ERROR: Unexpected error for '{image.name}': {e}")
        # save
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = self.output_dir / "labels.json"
        with output_file_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Labels saved to {output_file_path}")


def json_to_csv(json_file_path: Path) -> pd.DataFrame:
    """Converts a JSON file to pandas DataFrame."""
    with json_file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # dict key -> index
    df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    # df.to_csv(csv_file_path, index=True)
    return df


if __name__ == "__main__":
    context_prompt_file = Path("prompts") / "context_prompt.txt"
    input_dir = Path("frames")
    output_dir = Path("labels")

    # Instantiate the generator.
    label_generator = LabelGenerator(
        llm_deployment_name="o4-mini-labeling",
        context_prompt_file=context_prompt_file,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    # get the response from llm model
    label_generator.generate_json_from_multiple_images(max_workers=5)

    label_path = output_dir / "labels.json"
    df = json_to_csv(label_path)
    print(df)
