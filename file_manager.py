from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
import os
from azure.storage.filedatalake import DataLakeServiceClient, FileSystemClient
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()


@dataclass(frozen=True)
class AzureStorageConfig:
    """
    An immutable dataclass to store configuration for the Azure Storage service.
    It reads the connection string from an environment variable.
    """

    connection_string: str = os.getenv("CONNECTION_STRING")

    def __post_init__(self):

        # check if the connection string is valid
        if not self.connection_string:
            raise ValueError("CONNECTION_STRING environment variable is not set.")


class AzureStorageManager:
    """
    A client for interacting with Azure Data Lake Storage Gen2.
    This manager is designed to be stateless, meaning each method call is independent
    and does not rely on or modify the manager's internal state.
    """

    def __init__(self):
        """Initializes the storage manager and the main DataLakeServiceClient."""
        self.config = AzureStorageConfig()
        self.client = DataLakeServiceClient.from_connection_string(
            self.config.connection_string
        )
        logging.info("AzureStorageManager initialized successfully.")

    def get_container_client(
        self, container_name: str = "test", create_if_not_exists: bool = False
    ) -> FileSystemClient:
        """
        Retrieves a client for a specific container (file system).

        Args:
            container_name: The name of the container.
            create_if_not_exists: If True, creates the container if it doesn't exist.

        Returns:
            A FileSystemClient instance for the specified container.
        """
        container_client = self.client.get_file_system_client(
            file_system=container_name
        )
        # check if the container exists
        if create_if_not_exists and not container_client.exists():
            container_client.create_file_system()
            logging.info(f"Container {container_name} created.")
        return container_client

    def delete_container(self, container_name: str):
        """
        Deletes an entire container.

        Args:
            container_name: The name of the container to delete.
        """
        try:
            container_client = self.service_client.get_file_system_client(
                file_system=container_name
            )
            if container_client.exists():
                container_client.delete_file_system()
                logging.info(f"Container '{container_name}' deleted successfully.")
            else:
                logging.warning(f"Container '{container_name}' not found.")
        except Exception as e:
            logging.error(f"Error deleting container '{container_name}': {e}")
            raise

    def look_up_files_in_directory(
        self, azure_dir_to_search: str, container_name: str = "test"
    ):
        """
        Lists the files and subdirectories within a specified Azure directory.

        Args:
            azure_dir_to_search: The path of the directory to search.
            container_name: The name of the container.
        """
        try:
            container_client = self.get_container_client(container_name)
            paths = list(container_client.get_paths(path=azure_dir_to_search))

            if not paths:
                logging.warning(
                    f"No files or directories found in '{azure_dir_to_search}'."
                )
                return

            logging.info(f"Contents of '{azure_dir_to_search}':")
            for path in paths:
                item_type = "Directory" if path.is_directory else "File"
                logging.info(f" - Name: {path.name}, Type: {item_type}")
        except Exception as e:
            logging.error(f"Error listing files in '{azure_dir_to_search}': {e}")
            raise

    def delete_directory(self, azure_dir_to_delete: str, container_name: str = "test"):
        """
        Deletes a directory and all its contents recursively.

        Args:
            azure_dir_to_delete: The path of the directory to delete.
            container_name: The name of the container.
        """
        try:
            container_client = self.get_container_client(container_name)
            dir_client = container_client.get_directory_client(azure_dir_to_delete)
            if dir_client.exists():
                dir_client.delete_directory()
                logging.info(
                    f"Directory {azure_dir_to_delete} deleted from container {container_name}."
                )
            else:
                logging.warning(
                    f"Directory {azure_dir_to_delete} does not exist in container {container_name}."
                )
        except Exception as e:
            logging.error(
                f"Error deleting directory {azure_dir_to_delete} in container {container_name}: {e}"
            )
            raise

    def upload_files(
        self,
        local_files_dir: Path,
        azure_dir_to_upload: str,
        container_name: str = "test",
        pattern: str | None = None,
    ):
        """
        Uploads files from a local directory to a specified Azure directory.
        If a pattern is provided, only files matching the pattern will be uploaded.

        Args:
            local_files_dir: The local directory Path object containing files to upload.
            azure_dir_to_upload: The destination directory path in Azure.
            container_name: The name of the container.
            pattern: An optional glob pattern to filter files (e.g., "*.jpg", "data_*.csv").
                    If None, all files in the directory will be uploaded.
        """
        try:
            container_client = self.get_container_client(
                container_name, create_if_not_exists=True
            )
            directory_client = container_client.get_directory_client(
                azure_dir_to_upload
            )
            if not directory_client.exists():
                directory_client.create_directory()

            # Determine the glob pattern. If the user provides one, use it. Otherwise, default to '*'.
            glob_pattern = pattern if pattern else "*"

            # Find files using the determined pattern.
            files_to_upload = [
                p for p in local_files_dir.glob(glob_pattern) if p.is_file()
            ]

            if not files_to_upload:
                logging.warning(
                    f"No files matching pattern '{glob_pattern}' found in local directory '{local_files_dir}' to upload."
                )
                return

            for file_path in tqdm(
                files_to_upload,
                desc=f"Uploading '{glob_pattern}' files to '{azure_dir_to_upload}'",
            ):
                file_client = directory_client.get_file_client(file_path.name)
                with open(file_path, "rb") as data:
                    file_client.upload_data(data, overwrite=True)

            logging.info(
                f"Successfully uploaded {len(files_to_upload)} files to '{azure_dir_to_upload}'."
            )

        except Exception as e:
            logging.error(f"Error during file upload: {e}")
            raise

    def download_files(
        self,
        local_dir_to_save: Path,
        azure_dir_to_download: str,
        container_name: str = "test",
    ):
        """
        Downloads all files from an Azure directory to a local directory.

        Args:
            local_dir_to_save: The local directory Path object to save files into.
            azure_dir_to_download: The source directory path in Azure.
            container_name: The name of the container.
        """
        try:
            container_client = self.get_container_client(container_name)
            # Convert generator to a list to check if it's empty
            paths = list(container_client.get_paths(path=azure_dir_to_download))

            if not paths:
                logging.warning(
                    f"No files found in '{azure_dir_to_download}' to download."
                )
                return

            local_dir_to_save.mkdir(parents=True, exist_ok=True)
            for path in tqdm(paths, desc=f"Downloading from '{azure_dir_to_download}'"):
                # Ensure the item is a file, not a subdirectory
                if not path.is_directory:
                    file_name = Path(path.name).name
                    download_path = local_dir_to_save / file_name
                    file_client = container_client.get_file_client(path.name)
                    with open(download_path, "wb") as local_file:
                        download_stream = file_client.download_file()
                        local_file.write(download_stream.readall())

            logging.info(f"Successfully downloaded files to '{local_dir_to_save}'.")

        except Exception as e:
            logging.error(f"Error during file download: {e}")
            raise

    def delete_file(self, file_path_to_delete: str, container_name: str = "test"):
        """
        Deletes a single file from a container.

        Args:
            file_path_to_delete: The full path of the file to delete (e.g., 'dir/file.txt').
            container_name: The name of the container.
        """
        try:
            container_client = self.get_container_client(container_name)
            file_client = container_client.get_file_client(file_path_to_delete)

            if file_client.exists():
                file_client.delete_file()
                logging.info(f"File '{file_path_to_delete}' deleted successfully.")
            else:
                logging.warning(f"File '{file_path_to_delete}' not found.")
        except Exception as e:
            logging.error(f"Error deleting file '{file_path_to_delete}': {e}")
            raise


if __name__ == "__main__":
    manager = AzureStorageManager()

    # delete directory
    # manager.delete_directory(azure_dir_to_delete="labels", container_name="test")
    # upload files
    manager.upload_files(
        local_files_dir=Path("labels"),
        azure_dir_to_upload="labels",
        container_name="test",
        pattern="*.csv",
    )
