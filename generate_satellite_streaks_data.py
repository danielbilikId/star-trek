import numpy as np
import cv2
import os
from astropy.io import fits
from astropy.visualization import ZScaleInterval

def enhance_contrast(image):
    """
    Enhance the contrast of the image using adaptive histogram equalization.
    Args:
        image (numpy.ndarray): Input grayscale image.
    Returns:
        numpy.ndarray: Image with enhanced contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def generate_satellite_streaks(image, num_streaks=1):
    """
    Draw curved lines with small to moderate curvature on the image.
    Args:
        image (numpy.ndarray): The input image.
        num_streaks (int): Number of curved streaks to draw.
    Returns:
        numpy.ndarray: Image with curved streaks.
    """
    output = image.copy()
    height, width = image.shape

    for _ in range(num_streaks):
        # Randomize start and end points
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
        
        # Calculate the midpoint of the line
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        # Add a small random offset to the midpoint to create curvature
        offset_range = max(width, height) // 20  # Adjust for small to moderate curvature
        ctrl_x = mid_x + np.random.randint(-offset_range, offset_range)
        ctrl_y = mid_y + np.random.randint(-offset_range, offset_range)
        
        thickness = np.random.randint(2, 5)  # Make streaks thicker
        intensity = np.random.uniform(0.8, 1.0) * image.max()  # Increase brightness of streaks

        # Generate points along the Bezier curve
        num_points = 500  # More points for smoother curve
        curve_points = []
        for t in np.linspace(0, 1, num_points):
            # Quadratic Bezier formula
            x = int((1 - t) ** 2 * x1 + 2 * (1 - t) * t * ctrl_x + t ** 2 * x2)
            y = int((1 - t) ** 2 * y1 + 2 * (1 - t) * t * ctrl_y + t ** 2 * y2)
            curve_points.append((x, y))

        # Draw the curve
        for i in range(len(curve_points) - 1):
            cv2.line(output, curve_points[i], curve_points[i + 1], int(intensity), thickness)

    return output

def process_fits_and_generate_dataset(input_fits_path, output_dir, num_images=100,noise_std_dev=1):
    """
    Process FITS files and generate a dataset with and without satellite streaks.
    Args:
        input_fits_path (str): Path to the input FITS file.
        output_dir (str): Directory to save the augmented dataset.
        num_images (int): Total number of images to generate.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the FITS file
    with fits.open(input_fits_path) as hdul:
        original_data = hdul[0].data.astype(np.float32)
        header = hdul[0].header

    # Normalize the data for visualization purposes
    zscale = ZScaleInterval()
    mean = np.mean(original_data)
    original_data = original_data - mean
    
    for i in range(num_images):
        if i % 2 == 0:
            # Without satellite streak
            augmented_image = original_data
            label = 'no_satellite'
        else:
            # With satellite streak
            augmented_image = generate_satellite_streaks(original_data, num_streaks=np.random.randint(1, 3))
            label = 'with_satellite'
            
        noise = np.random.normal(loc=0, scale=noise_std_dev, size=augmented_image.shape)
        augmented_image += noise
        # Save the augmented image as a FITS file
        filename = input_fits_path.split("/")[-1]
        filename = filename.split(".")[-2]
        output_fits_path = os.path.join(output_dir, f"{filename}_{label}_{i:03d}.fits")
        fits.writeto(output_fits_path, augmented_image, header, overwrite=True)
        
        print(f"Saved {output_fits_path}")
