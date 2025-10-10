import cv2
from pathlib import Path


# Function to extract frames from a video
# Python 3.10+ 에서는 | 를 사용, 이전 버전에서는 from typing import Optional 을 쓰고 Optional[int] 사용
def extract_frames(
    video_path: Path, output_dir: Path, duration: int | None = None
) -> None:
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # check if the video file is opened
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"FPS: {fps}")

    # create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    success, img = cap.read()
    # 1초당 1장 추출 : fps = 25 (1초동안 25개의 프레임으로 구성됨) -> 0 % 25 = 0, 25 % 25 = 0, 50 % 25 = 0, ...
    while success:
        if frame_count % fps == 0:
            frame_time = (
                frame_count // fps
            )  # 0 //25 = 0, 25 // 25 = 1, 50 // 25 = 2, ...
            cv2.imwrite(f"{output_dir}/frame_{frame_time}.jpg", img)

            if duration is not None and frame_time >= duration:
                print(f"Reached duration limit of {duration} seconds. Stopping.")
                break
        # read the next frame and update img variable
        success, img = cap.read()
        frame_count += 1

    # close the video file
    cap.release()
    print("Finished extracting frames.")


if __name__ == "__main__":
    video_path = Path("example_video_25fps.mp4")
    output_dir = Path("frames")
    extract_frames(video_path, output_dir, duration=5)
