import os
import random
import io
import scipy.io
import numpy as np
from PIL import Image

def load_unseen_quantization_matrices(mat_file_path):
    """
    Loads quantization matrices from a .mat file, handles MATLAB cell arrays 
    and 3D matrices, flattens them, and formats them for PIL.
    """
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"Could not find the .mat file at: {mat_file_path}")

    # Load the MATLAB file
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # Exclude standard metadata keys
    valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    print(f"Found the following data keys in the .mat file: {valid_keys}")
    
    if not valid_keys:
        raise ValueError("No valid data matrices found in the .mat file.")
    
    matrix_key = valid_keys[0]
    print(f"Extracting matrices using key: '{matrix_key}'")
    
    raw_matrices = mat_data[matrix_key]
    formatted_matrices = []

    # SCENARIO A: MATLAB Cell Array (Loaded as numpy object array)
    if raw_matrices.dtype == object:
        # Flatten the outer object array to loop through the individual cells
        for cell in raw_matrices.flatten():
            # Extract the actual numeric array inside the cell and flatten it
            num_array = np.array(cell).flatten()
            if len(num_array) == 64:
                formatted_matrices.append([int(x) for x in num_array])
                
    # SCENARIO B: Standard Numeric Array
    else:
        # If MATLAB saved it as a 3D matrix of (Rows, Columns, Depth) like (8, 8, N)
        if raw_matrices.ndim == 3 and raw_matrices.shape[0] == 8 and raw_matrices.shape[1] == 8:
            # Transpose it to (N, 8, 8) so we can iterate over the N matrices
            raw_matrices = np.transpose(raw_matrices, (2, 0, 1))
            
        for q_matrix in raw_matrices:
            flat_matrix = np.array(q_matrix).flatten()
            if len(flat_matrix) == 64:
                formatted_matrices.append([int(x) for x in flat_matrix])

    print(f"Successfully loaded and formatted {len(formatted_matrices)} valid matrices.")
    
    # Replicate the paper's experiment by slicing exactly 504 test matrices
    return formatted_matrices[:504]


def process_image(image_path, output_dir, test_matrices):
    """
    Extracts 256x256 non-overlapping patches from an image and generates 
    single and double compressed versions using random unseen matrices.
    """
    filename = os.path.basename(image_path).split('.')[0]
    
    try:
        # Convert to 'L' (Luminance/Grayscale) as the paper states compression is on the Y channel
        img = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return

    width, height = img.size
    patch_count = 0
    
    # Extract 256x256 non-overlapping patches
    for y in range(0, height, 256):
        for x in range(0, width, 256):
            
            # Ensure the patch is exactly 256x256 (skip edges if they are smaller)
            if x + 256 <= width and y + 256 <= height:
                patch = img.crop((x, y, x + 256, y + 256))
                
                # --- SINGLE COMPRESSION ---
                # Randomly select one quantization matrix from the pool
                q_single = random.choice(test_matrices)
                single_comp_path = os.path.join(output_dir, "single", f"{filename}_patch{patch_count}_single.jpg")
                
                # Save patch using the custom quantization matrix (PIL expects a dict for qtables)
                patch.save(single_comp_path, "JPEG", qtables={0: q_single})
                
                # --- DOUBLE COMPRESSION ---
                # Randomly select Q1 and Q2 from the pool, ensuring Q1 != Q2
                q1, q2 = random.sample(test_matrices, 2)
                double_comp_path = os.path.join(output_dir, "double", f"{filename}_patch{patch_count}_double.jpg")
                
                # Step 1: Compress with Q1 to an in-memory buffer
                buffer = io.BytesIO()
                patch.save(buffer, format="JPEG", qtables={0: q1})
                
                # Step 2: Decompress (read back from buffer)
                buffer.seek(0)
                temp_img = Image.open(buffer)
                
                # Step 3: Re-compress with Q2 and save to disk
                temp_img.save(double_comp_path, "JPEG", qtables={0: q2})
                
                patch_count += 1


def main():
    # --- CONFIGURATION (Change these paths to match your system) ---
    input_image_dir = "/Users/shivamsingh/Desktop/DIP/UCID1338"  # Folder with your raw TIF/BMP/PNG images
    output_dir = "/Users/shivamsingh/Desktop/DIP/UCID_dataset"               # Where the patched JPEGs will go
    mat_file_path = "/Users/shivamsingh/Desktop/DIP/q_table_list.mat"                 # Path to your .mat file
    # ---------------------------------------------------------------
    
    # Create the output directories if they don't exist
    os.makedirs(os.path.join(output_dir, "single"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "double"), exist_ok=True)
    
    # 1. Load the matrices from the .mat file
    try:
        test_matrices = load_unseen_quantization_matrices(mat_file_path)
    except Exception as e:
        print(f"Failed to load matrices: {e}")
        return
        
    if not test_matrices:
        print("No valid matrices to process. Exiting.")
        return

    # 2. Find all images in the input directory
    if not os.path.exists(input_image_dir):
        print(f"Input directory '{input_image_dir}' does not exist.")
        return

    valid_extensions = ('.tif', '.tiff', '.png', '.bmp', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(input_image_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No images found in {input_image_dir}.")
        return

    print(f"\nFound {len(image_files)} images. Starting patching and compression...")
    
    # 3. Process each image
    for img_file in image_files:
        img_path = os.path.join(input_image_dir, img_file)
        process_image(img_path, output_dir, test_matrices)
        
    print("\nDataset preparation complete. Check the output folders!")

if __name__ == "__main__":
    main()