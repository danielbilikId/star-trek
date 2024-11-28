import os
import shutil

# Define source and destination directories
source_folder = "D:\LAST"
destination_folder = "./weizmann_telescope/data"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Iterate over all files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file ends with "_Image_1.fits"
    if filename.endswith("_Image_1.fits"):
        # Construct full file paths
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        
        # Copy the file to the destination folder
        shutil.copy2(source_file, destination_file)
        print(f"Copied: {filename} to {destination_folder}")

print("All matching files have been copied.")
