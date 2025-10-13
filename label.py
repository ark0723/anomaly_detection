import base64
from openai import OpenAIError  # Import specific error for better handling
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
from datetime import datetime

from llm import LabelLLM


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
        self.cache_dir = self.output_dir / "cache"
        self._validate_path()

        # On startup, gather cached results from any previous failed run for resiliency.
        if self.cache_dir.exists() and any(self.cache_dir.iterdir()):
            print(
                "💡 INFO: Leftover cache found. Gathering results from previous run..."
            )
            self.gather_json_from_cache()

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

        # Parse the response
        # [Cache]: Save the JSON response to a file
        parsed_response = self.parse_json_str(raw_response)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # image_path.stem: file name without extension
        cache_file_path = self.cache_dir / f"{image_path.stem}.json"
        with cache_file_path.open("w", encoding="utf-8") as f:
            json.dump(parsed_response, f, indent=4, ensure_ascii=False)
        return parsed_response

    def _get_processed_image_set(self) -> set:
        """return a set of image names that have been processed from the output directory."""
        processed_image_set = set()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for output_file in self.output_dir.glob("*.json"):
            with output_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    processed_image_set.update(data.keys())
                except json.JSONDecodeError:
                    print(f"❌ ERROR: Failed to load JSON file from '{output_file}'.")
        return processed_image_set

    def generate_json_from_multiple_images(self, max_workers: int = 10) -> None:
        """Generates JSON labels for multiple images concurrently using a thread pool.

        This method scans an input directory for image files, submits each image
        to a ThreadPoolExecutor for processing, and collects the resulting
        JSON data in the order the images were submitted.

        Args:
            max_workers (int): The maximum number of worker threads to use for
                concurrent processing. Defaults to 10.

        Returns:
            None
        """
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
        # 이미 처리된 이미지 이름의 Set
        processed_image_set = self._get_processed_image_set()

        images = list(self.input_dir.glob("*.*"))
        images = [
            img
            for img in images
            if img.stem not in processed_image_set
            and img.suffix.lower() in image_extensions
        ]
        if not images:
            print(f"No new images found in {self.input_dir}")
            return

        print(f"INFO: Found {len(images)} new images to process.")
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
                    future.result()
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

        # [cache]: load the JSON responses from the files
        print("INFO: All images processed. Gathering results...")
        self.gather_json_from_cache()

    def gather_json_from_cache(self) -> None:
        """Gathers JSON responses from the cache directory, saves to a final file, and robustly cleans up the cache."""
        cache_files = sorted(self.cache_dir.glob("*.json"))

        if not cache_files:
            print("INFO: No cache files found. Nothing to gather.")
            return

        results = OrderedDict()

        for cache_file in cache_files:
            try:
                with cache_file.open("r", encoding="utf-8") as f:
                    # Object_pairs_hook=OrderedDict: Maintain the order of the items in the JSON file
                    data = json.load(f, object_pairs_hook=OrderedDict)
                    results[cache_file.stem] = data
            except Exception as e:
                print(f"❌ ERROR: Failed to load JSON file from '{cache_file}': {e}")
            # exception is not raised
            else:
                # delete the cache file
                cache_file.unlink()

        # save the results to a file
        timestamp = datetime.now().isoformat("T", "seconds").replace(":", "-")
        output_file_path = self.output_dir / f"labels_{timestamp}.json"
        with output_file_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Labels saved to {output_file_path}")
        self.cache_dir.rmdir()


def json_to_df(json_file_path: Path) -> pd.DataFrame:
    """Converts a JSON file to pandas DataFrame."""
    try:
        with json_file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # dict key -> index
        df = pd.DataFrame.from_dict(data, orient="index").sort_index()
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {json_file_path}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"❌ ERROR: Failed to decode JSON file: {json_file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        return pd.DataFrame()


def combine_jsons_to_single_df(
    json_dir_path: Path, csv_file_path: Path | None = None
) -> pd.DataFrame:
    """
    Finds all JSON files in a directory, reads them into DataFrames,
    and concatenates them into a single DataFrame.
    """
    # 1. find all json files in the directory
    json_files = list(json_dir_path.glob("*.json"))

    if not json_files:
        print(f"⚠️ No JSON files found in {json_dir_path}")
        # return an empty dataframe
        return pd.DataFrame()

    # 2. convert json files to a dataframe list
    df_list = []
    for file in json_files:
        df = json_to_df(file)
        if not df.empty:
            df_list.append(df)

    if not df_list:
        print("INFO: No valid data could be loaded from any JSON file.")
        return pd.DataFrame()

    # 3. sort the dataframe
    combined_df = pd.concat(df_list).sort_index()
    if csv_file_path:
        combined_df.to_csv(csv_file_path, index=True)
        print(f"Combined Data (csv) saved to {csv_file_path}")

    return combined_df


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
    # label_generator.generate_json_from_multiple_images(max_workers=5)

    df = combine_jsons_to_single_df(
        json_dir_path=output_dir, csv_file_path=output_dir / "labels.csv"
    )
    print(df)
