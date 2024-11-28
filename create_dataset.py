import os
from config import IMAGE_PATH
from generate_satellite_streaks_data import process_fits_and_generate_dataset

input_fits_path = IMAGE_PATH
output_dir = "augmented_dataset"

for image in os.listdir(input_fits_path):
    process_fits_and_generate_dataset(f'./data/{image}', output_dir, num_images=10)

