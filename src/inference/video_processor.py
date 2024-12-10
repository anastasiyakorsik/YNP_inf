import time
import os
import subprocess

def get_frame_times(video_path):
    """
    Gets time for each video frame

    Arguments:
        video_path (str): Path to video

    Returns:
        frame_times: Array of each frame time
    """

    # "frame=pkt_dts_time" - variant 1
    # "frame=pkt_pts_time" - var 2
    # "frame=best_effort_timestamp_time"
    command = [
        "ffprobe",
        "-show_frames",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv",
        video_path
    ]

    # Run cmd by subprocess
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"FFprobe error: {result.stderr}")

    # Parse result
    frame_times = []
    for line in result.stdout.splitlines():
        if line.startswith("frame"):
            time = line.split(",")[1]
            if 'side' in line:
                time = time.split("s")[0]
            frame_times.append(float(time))
    
    return frame_times

def check_video_extension(video_path):
    """
    Checks video extension for compliance with the specified ones
    """
    valid_extensions = {"avi", "mp4", "m4v", "mov", "mpg", "mpeg", "wmv"}
    ext = os.path.splitext(video_path)[1][1:].lower()
    return ext in valid_extensions
