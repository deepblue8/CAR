import os
import shutil

def delete_files_in_subdirectories(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for directory in dirs:
            current_dir = os.path.join(root, directory)

            frame_folder = os.path.join(current_dir, "frame")
            if os.path.exists(frame_folder):
                shutil.rmtree(frame_folder)

            another_file = os.path.join(current_dir, "MOT16-03-results.mp4")
            if os.path.exists(another_file):
                os.remove(another_file)

root_folder_path = "E://VLBNewLabel"
delete_files_in_subdirectories(root_folder_path)
