import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np
from io import BytesIO
import os
import time
import json
import base64
import tempfile
import scipy.fft as fft

# Attempt to import TensorFlow, pyexiftool, and scikit-image
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not found. Deepfake detection will not be available.")

try:
    import exiftool
    EXIFTOOL_AVAILABLE = True
except ImportError:
    EXIFTOOL_AVAILABLE = False
    print("pyexiftool not found. If installed, ensure ExifTool executable is also available.")

try:
    from skimage.restoration import estimate_sigma
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("scikit-image not found. Noise analysis will be limited.")

# --- Configuration ---
# Path to the CNN model weights
MODEL_PATH = r"D:\Forensics HD\mesonet_final.h5"

# Path to exiftool executable
EXIFTOOL_EXECUTABLE = r"D:\Forensics HD\exiftool.exe"

IMG_SIZE = (128, 128)  # Adjusted to reduce feature map size
DEEPFAKE_THRESHOLD = 0.7  # Adjusted based on test metrics

# --- Deepfake Detector Functions ---
deepfake_model_loaded = None

def debug_feature_map_shape(model, input_array):
    """Debug function to trace the shape before the dense layer."""
    try:
        for layer in model.layers:
            input_array = layer(input_array)
            print(f"Layer {layer.name}: Output shape = {input_array.shape}")
        return input_array
    except Exception as e:
        print(f"Error tracing feature map shapes: {e}")
        return None

def load_deepfake_model_tf():
    global deepfake_model_loaded
    if not TF_AVAILABLE:
        raise ValueError("TensorFlow is not available, cannot load CNN model.")

    if MODEL_PATH and tf.io.gfile.exists(MODEL_PATH):
        try:
            deepfake_model_loaded = tf.keras.models.load_model(MODEL_PATH)
            print(f"CNN model loaded successfully from {MODEL_PATH}")
            # Debug: Print model input shape
            input_shape = deepfake_model_loaded.input_shape
            print(f"Model expected input shape: {input_shape}")
        except Exception as e:
            raise ValueError(f"Error loading CNN model from {MODEL_PATH}: {e}")
    else:
        raise ValueError(f"Model path {MODEL_PATH} not found.")
    return deepfake_model_loaded

# Load the model when the script starts
try:
    load_deepfake_model_tf()
except ValueError as e:
    print(e)
    deepfake_model_loaded = None

def predict_deepfake_tf(image_pil):
    global deepfake_model_loaded
    if not TF_AVAILABLE or deepfake_model_loaded is None:
        raise ValueError("CNN model is not loaded or TensorFlow is not available.")

    try:
        img = image_pil.convert('RGB').resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        print(f"Input shape to model: {img_array.shape}")  # Debug print
        
        # Debug: Trace feature map shapes
        debug_feature_map_shape(deepfake_model_loaded, img_array)
        
        prediction = deepfake_model_loaded.predict(img_array)
        
        # Handle sigmoid output (single probability)
        if prediction.shape[-1] == 1:  # Sigmoid output
            score = float(prediction[0][0])
            print(f"Sigmoid output detected. Raw score: {score}")
        else:  # Softmax output (two classes: authentic, deepfake)
            score = tf.nn.softmax(prediction[0])[1].numpy()  # Probability for 'deepfake' class
            print(f"Softmax output detected. Deepfake score: {score}")
        
        # Adjust based on class indices: Fake=0, Real=1
        is_real = score > DEEPFAKE_THRESHOLD  # score > threshold means class 1 (Real)
        is_deepfake = not is_real  # Invert: Deepfake if not Real
        heatmap = None  # Grad-CAM or similar would go here
        print(f"Final score: {score}, Classified as: {'Real' if is_real else 'Fake'}")
        return is_deepfake, score, heatmap
    except Exception as e:
        raise ValueError(f"Error during CNN prediction: {e}")

# --- Hex Dump Function ---
def generate_hex_dump(file_bytes):
    """
    Generates a hex dump of the entire file bytes, similar to HxD.
    
    Args:
        file_bytes (bytes): The binary content of the file.
    
    Returns:
        str: A string containing the hex dump.
    """
    hex_lines = []
    
    for i in range(0, len(file_bytes), 16):
        # Get a chunk of 16 bytes
        chunk = file_bytes[i:i + 16]
        # Offset in hex
        offset = f"{i:08X}"
        # Hex representation
        hex_values = [f"{byte:02X}" for byte in chunk]
        # Pad with spaces if the chunk is less than 16 bytes
        hex_values.extend(['  '] * (16 - len(chunk)))
        hex_str = ' '.join(hex_values)
        # ASCII representation (printable characters or '.')
        ascii_str = ''.join(chr(byte) if 32 <= byte <= 126 else '.' for byte in chunk)
        # Format the line
        hex_lines.append(f"{offset}  {hex_str}  {ascii_str}")
    
    return '\n'.join(hex_lines)

# --- Statistical Analysis Function ---
def perform_statistical_analysis(image_pil):
    """
    Performs statistical analysis on the image by analyzing color histograms and DCT coefficients.
    
    Args:
        image_pil (PIL.Image): Input image.
    
    Returns:
        dict: Results containing analysis of histograms and DCT coefficients.
    """
    try:
        # Convert image to numpy array
        image_array = np.array(image_pil.convert('RGB'))
        
        # --- Color Histogram Analysis ---
        # Compute histograms for R, G, B channels
        hist_r = cv2.calcHist([image_array], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([image_array], [1], None, [256], [0, 256]).flatten()
        hist_b = cv2.calcHist([image_array], [2], None, [256], [0, 256]).flatten()
        
        # Analyze histogram for anomalies (e.g., high variance or discontinuities)
        hist_variance_r = np.var(hist_r)
        hist_variance_g = np.var(hist_g)
        hist_variance_b = np.var(hist_b)
        
        # Threshold for detecting anomalies (high variance might indicate manipulation)
        variance_threshold = 1e6  # Adjust based on empirical testing
        histogram_anomaly = (
            hist_variance_r > variance_threshold or
            hist_variance_g > variance_threshold or
            hist_variance_b > variance_threshold
        )
        
        # --- DCT Coefficient Analysis ---
        # Convert to grayscale for DCT
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32)
        
        # Apply DCT (block-wise, 8x8 blocks)
        height, width = gray.shape
        dct_coeffs = np.zeros_like(gray)
        block_size = 8
        for i in range(0, height - block_size + 1, block_size):
            for j in range(0, width - block_size + 1, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                if block.shape == (block_size, block_size):
                    dct_block = fft.dctn(block, norm='ortho')
                    dct_coeffs[i:i+block_size, j:j+block_size] = dct_block
        
        # Analyze DCT coefficients (e.g., high-frequency components)
        high_freq_energy = np.sum(np.abs(dct_coeffs[1:, 1:]))  # Exclude DC component
        total_energy = np.sum(np.abs(dct_coeffs))
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Threshold for high-frequency energy (too high might indicate tampering)
        high_freq_threshold = 0.9  # Adjust based on empirical testing
        dct_anomaly = high_freq_ratio > high_freq_threshold
        
        # Prepare results
        results = {
            "histogram_variance": {
                "R": float(hist_variance_r),
                "G": float(hist_variance_g),
                "B": float(hist_variance_b)
            },
            "histogram_anomaly_detected": histogram_anomaly,
            "dct_high_freq_ratio": float(high_freq_ratio),
            "dct_anomaly_detected": dct_anomaly,
            "overall_anomaly_detected": histogram_anomaly or dct_anomaly
        }
        
        return results
    
    except Exception as e:
        print(f"Error in statistical analysis: {e}")
        return {
            "error": f"Error: {str(e)}",
            "histogram_variance": None,
            "histogram_anomaly_detected": False,
            "dct_high_freq_ratio": None,
            "dct_anomaly_detected": False,
            "overall_anomaly_detected": False
        }

# --- Blending Analysis Function ---
def detect_blending(image_pil):
    """
    Detects blended regions in the image by analyzing edges and gradients.
    
    Args:
        image_pil (PIL.Image): Input image.
    
    Returns:
        dict: Results containing blending detection status and marked image.
    """
    try:
        # Convert PIL image to OpenCV format
        image_array = np.array(image_pil.convert('RGB'))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Edge detection using Canny
        edges = cv2.Canny(gray, 100, 200)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold gradients to find sharp transitions
        grad_threshold = np.percentile(grad_magnitude, 95)  # Top 5% of gradients
        sharp_transitions = grad_magnitude > grad_threshold
        
        # Find regions with high edge density and sharp transitions
        kernel = np.ones((5, 5), np.uint8)
        edge_density = cv2.dilate(edges, kernel, iterations=1)
        blending_mask = (edge_density > 0) & sharp_transitions
        
        # Check if significant blending regions are detected
        blending_area = np.sum(blending_mask)
        total_area = blending_mask.size
        blending_ratio = blending_area / total_area if total_area > 0 else 0
        
        # Threshold for detecting blending (adjust based on empirical testing)
        blending_threshold = 0.01  # 1% of the image area
        blending_detected = blending_ratio > blending_threshold
        
        # Mark suspected regions on the image
        marked_image = image_array.copy()
        if blending_detected:
            # Convert mask to 3 channels for overlay
            blending_mask_3ch = np.zeros_like(image_array)
            blending_mask_3ch[:, :, 1] = blending_mask.astype(np.uint8) * 255  # Green channel
            marked_image = cv2.addWeighted(marked_image, 0.7, blending_mask_3ch, 0.3, 0)
        
        # Convert marked image to bytes
        _, buffer = cv2.imencode('.png', marked_image)
        marked_image_bytes = buffer.tobytes()
        
        return {
            "blending_detected": blending_detected,
            "blending_ratio": float(blending_ratio),
            "details": f"Blending ratio: {blending_ratio:.4f}",
            "marked_image_bytes": marked_image_bytes if blending_detected else None
        }
    
    except Exception as e:
        print(f"Error in blending analysis: {e}")
        return {
            "blending_detected": False,
            "blending_ratio": 0.0,
            "details": f"Error: {str(e)}",
            "marked_image_bytes": None
        }

# --- Enhancement Detection Function ---
def detect_enhancements(image_pil):
    """
    Detects post-processing enhancements like sharpening and noise reduction.
    
    Args:
        image_pil (PIL.Image): Input image.
    
    Returns:
        dict: Results containing enhancement detection status.
    """
    try:
        # Convert to OpenCV format
        image_array = np.array(image_pil.convert('RGB'))
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # --- Detect Sharpening (High-frequency components) ---
        # Use Laplacian to detect over-sharpened edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        # Threshold for detecting sharpening (adjust based on empirical testing)
        sharpening_threshold = 1000  # High variance indicates possible sharpening
        sharpening_detected = laplacian_var > sharpening_threshold
        
        # --- Detect Noise Reduction (Overly smooth regions) ---
        # Compute local variance in small patches
        patch_size = 16
        height, width = gray.shape
        local_variances = []
        for i in range(0, height - patch_size + 1, patch_size):
            for j in range(0, width - patch_size + 1, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                if patch.shape == (patch_size, patch_size):
                    local_variances.append(np.var(patch))
        
        # Analyze variance distribution
        avg_variance = np.mean(local_variances) if local_variances else 0
        low_variance_ratio = sum(1 for var in local_variances if var < 10) / len(local_variances) if local_variances else 0
        
        # Threshold for detecting noise reduction (too many low-variance patches)
        noise_reduction_threshold = 0.5  # 50% of patches with low variance
        noise_reduction_detected = low_variance_ratio > noise_reduction_threshold
        
        return {
            "sharpening_detected": sharpening_detected,
            "sharpening_laplacian_variance": float(laplacian_var),
            "noise_reduction_detected": noise_reduction_detected,
            "noise_reduction_low_variance_ratio": float(low_variance_ratio),
            "details": f"Sharpening Laplacian variance: {laplacian_var:.2f}, Low variance ratio: {low_variance_ratio:.4f}"
        }
    
    except Exception as e:
        print(f"Error in enhancement detection: {e}")
        return {
            "sharpening_detected": False,
            "sharpening_laplacian_variance": 0.0,
            "noise_reduction_detected": False,
            "noise_reduction_low_variance_ratio": 0.0,
            "details": f"Error: {str(e)}"
        }

# --- Image Forensics Functions ---
def perform_ela(pil_image, quality=90, scale=10):
    pil_image = pil_image.convert('RGB')
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    resaved_image = Image.open(buffer)
    ela_image = ImageChops.difference(pil_image, resaved_image)
    extrema = ela_image.getextrema()
    max_diff = 0
    for i in range(len(extrema)):
        if isinstance(extrema[i], tuple):
            max_diff = max(max_diff, extrema[i][1])
        else:
            max_diff = max(max_diff, extrema[1])
    if max_diff == 0: max_diff = 1
    scale_factor = min(255.0 / max_diff * scale, scale * 10)
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor)
    
    # Compute variance of the ELA image
    ela_array = np.array(ela_image)
    ela_variance = np.var(ela_array)
    
    return ela_image, ela_variance

def analyze_noise_basic(image_array):
    if not SKIMAGE_AVAILABLE:
        return {"estimated_noise_sigma": "scikit-image not available"}
    
    try:
        if image_array.ndim == 3 and image_array.shape[2] == 4: # RGBA
            image_array = image_array[:, :, :3] # Convert to RGB
        
        if image_array.dtype != np.float64 and image_array.dtype != np.float32:
            image_array_float = image_array.astype(np.float32) / 255.0
        else:
            image_array_float = image_array

        sigma_est = estimate_sigma(image_array_float, channel_axis=-1, average_sigmas=True)
        return {"estimated_noise_sigma": float(sigma_est)}
    except Exception as e:
        print(f"Error in noise analysis: {e}")
        return {"estimated_noise_sigma": f"Error: {str(e)}"}

def detect_copy_move_forgery(image_pil, block_size=16, similarity_threshold=0.95):
    """
    Detects copy-move forgery in an image by identifying duplicated regions.
    
    Args:
        image_pil (PIL.Image): Input image.
        block_size (int): Size of the blocks to compare (e.g., 16x16 pixels).
        similarity_threshold (float): Threshold for block similarity (0 to 1).
    
    Returns:
        dict: Results containing tampering status, marked image (if tampering detected), and details.
    """
    try:
        # Convert PIL image to OpenCV format (numpy array)
        image_array = np.array(image_pil.convert('RGB'))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for simpler processing
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Ensure image dimensions are sufficient for block processing
        if height < block_size or width < block_size:
            return {
                "tampering_detected": False,
                "details": "Image too small for copy-move analysis.",
                "marked_image_bytes": None
            }
        
        # Divide image into overlapping blocks
        blocks = []
        positions = []
        for y in range(0, height - block_size + 1, 4):  # Step of 4 to reduce computation
            for x in range(0, width - block_size + 1, 4):
                block = gray[y:y+block_size, x:x+block_size]
                if block.shape == (block_size, block_size):  # Ensure block is correct size
                    # Use average intensity as a simple feature
                    block_feature = np.mean(block)
                    blocks.append(block_feature)
                    positions.append((x, y))
        
        blocks = np.array(blocks)
        num_blocks = len(blocks)
        
        # Compare blocks to find duplicates
        duplicated_pairs = []
        for i in range(num_blocks):
            for j in range(i + 1, num_blocks):
                # Compute similarity (normalized difference)
                similarity = 1.0 - abs(blocks[i] - blocks[j]) / 255.0
                if similarity > similarity_threshold:
                    # Ensure blocks are not too close (avoid false positives from nearby regions)
                    pos1, pos2 = positions[i], positions[j]
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    if distance > block_size * 2:  # Minimum distance between duplicated regions
                        duplicated_pairs.append((pos1, pos2))
        
        # If duplicated regions are found, mark them on the image
        if duplicated_pairs:
            marked_image = image_array.copy()
            for (x1, y1), (x2, y2) in duplicated_pairs:
                # Draw rectangles around duplicated regions
                cv2.rectangle(marked_image, (x1, y1), (x1 + block_size, y1 + block_size), (0, 0, 255), 2)
                cv2.rectangle(marked_image, (x2, y2), (x2 + block_size, y2 + block_size), (0, 0, 255), 2)
                # Draw a line connecting the regions
                cv2.line(marked_image, (x1 + block_size//2, y1 + block_size//2),
                         (x2 + block_size//2, y2 + block_size//2), (0, 255, 0), 2)
            
            # Convert marked image to bytes for display
            _, buffer = cv2.imencode('.png', marked_image)
            marked_image_bytes = buffer.tobytes()
            
            return {
                "tampering_detected": True,
                "details": f"Found {len(duplicated_pairs)} duplicated region(s).",
                "marked_image_bytes": marked_image_bytes
            }
        else:
            return {
                "tampering_detected": False,
                "details": "No duplicated regions detected.",
                "marked_image_bytes": None
            }
    except Exception as e:
        print(f"Error in copy-move forgery detection: {e}")
        return {
            "tampering_detected": False,
            "details": f"Error: {str(e)}",
            "marked_image_bytes": None
        }

# --- Metadata Analysis Functions ---
def extract_metadata_with_exiftool(file_bytes, filename):
    global EXIFTOOL_EXECUTABLE
    metadata_dict = {"Filename": filename, "File Size (bytes)": len(file_bytes)}
    
    if not EXIFTOOL_AVAILABLE:
        metadata_dict["ExifTool Status"] = "pyexiftool library not installed."
    elif EXIFTOOL_EXECUTABLE and not os.path.exists(EXIFTOOL_EXECUTABLE):
        metadata_dict["ExifTool Status"] = f"Specified ExifTool executable not found at: {EXIFTOOL_EXECUTABLE}. pyexiftool will try system PATH."
        current_exiftool_executable_path = None 
    else:
        current_exiftool_executable_path = EXIFTOOL_EXECUTABLE

    if not EXIFTOOL_AVAILABLE:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.webp')):
            try:
                img = Image.open(BytesIO(file_bytes))
                if hasattr(img, '_getexif'):
                    exif_data = img._getexif()
                    if exif_data:
                        from PIL.ExifTags import TAGS
                        metadata_dict["Basic EXIF (Pillow)"] = {TAGS.get(tag, tag): str(value) for tag, value in exif_data.items()}
            except Exception as pil_e:
                metadata_dict["Pillow EXIF Error"] = str(pil_e)
        return metadata_dict

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(file_bytes)
            temp_file_path = tmp.name
        
        with exiftool.ExifToolHelper(executable=current_exiftool_executable_path) as et:
            raw_metadata_list = et.get_metadata(temp_file_path)
            if raw_metadata_list and isinstance(raw_metadata_list, list):
                raw_metadata = raw_metadata_list[0]
                for k, v in raw_metadata.items():
                    metadata_dict[k] = str(v)
            else:
                metadata_dict["ExifTool Status"] = "No metadata retrieved or unexpected format from ExifTool."
                
    except exiftool.exceptions.ExifToolExecuteError as ete:
        err_msg = str(ete)
        if "NewValue" in err_msg:
             err_msg += " (This might indicate ExifTool executable was not found or is not working. Check PATH or EXIFTOOL_EXECUTABLE setting.)"
        metadata_dict["ExifTool Execution Error"] = err_msg
        print(f"ExifTool Execution Error: {err_msg}")
    except Exception as e:
        metadata_dict["ExifTool Status"] = f"Generic error during ExifTool processing: {str(e)}"
        print(f"Generic ExifTool Error: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e_rem:
                print(f"Error removing temp file {temp_file_path}: {e_rem}")
    
    return metadata_dict

# --- Report Generation Function ---
def generate_report_html(results):
    html_content = """
    <html><head><title>Forensic Report</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; }
        h1, h2, h3 { color: #1a237e; }
        h1 { text-align: center; border-bottom: 2px solid #1a237e; padding-bottom: 10px; }
        pre { background-color: #e8eaf6; padding: 15px; border: 1px solid #c5cae9; 
              overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; border-radius: 5px; font-size: 0.9em; }
        .section { margin-bottom: 25px; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-deepfake { color: #c62828; font-weight: bold; }
        .status-authentic { color: #2e7d32; font-weight: bold; }
        .status-tampered { color: #c62828; font-weight: bold; }
        .status-not-tampered { color: #2e7d32; font-weight: bold; }
        .status-anomaly { color: #c62828; font-weight: bold; }
        .status-no-anomaly { color: #2e7d32; font-weight: bold; }
        .status-enhanced { color: #c62828; font-weight: bold; }
        .status-not-enhanced { color: #2e7d32; font-weight: bold; }
        img.report-image { max-width: 100%; height: auto; border: 1px solid #ccc; margin-top:10px; border-radius: 4px; display: block; margin-left: auto; margin-right: auto; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #3949ab; color: white; }
        p.explanation { font-style: italic; color: #555; margin-top: 10px; }
    </style>
    </head><body>
    <h1>Digital Media Forensic Report</h1>
    """
    if results.get("filename"):
        html_content += f"<div class='section'><h2>File: {results['filename']}</h2></div>"

    if "deepfake_prediction" in results:
        pred = results["deepfake_prediction"]
        status_class = "status-deepfake" if pred["is_deepfake"] else "status-authentic"
        status_text = "Deepfake" if pred["is_deepfake"] else "Authentic"
        html_content += f"<div class='section'><h3>Deepfake Detection</h3>"
        confidence = pred.get("confidence_displayed", (1.0 - pred["score"]) if pred["is_deepfake"] else pred["score"])
        label = "Deepfake Confidence" if pred["is_deepfake"] else "Confidence"
        html_content += f"<p>Status: <span class='{status_class}'>{status_text}</span> ({label}: {confidence:.2f})</p>"

        # Dynamic explanation for Deepfake Detection
        if "frame_details" in pred:  # Video analysis
            num_frames = len(pred["frame_details"])
            deepfake_frames = sum(1 for fd in pred["frame_details"] if fd["is_deepfake"])
            explanation = f"This video was analyzed frame by frame to check for signs of artificial manipulation, like those created by deepfake technology. We looked at {num_frames} frames, and {deepfake_frames} of them showed signs of being manipulated. "
            if pred["is_deepfake"]:
                explanation += f"Overall, the video appears to have been altered, with a {confidence:.2f} confidence score indicating it's likely a deepfake. This means there's a strong chance this video isn't genuine and may have been created or edited using AI techniques."
            else:
                explanation += f"Overall, the video appears to be genuine, with a {confidence:.2f} confidence score suggesting it's authentic. This means it's likely this video hasn't been manipulated by deepfake methods."
        else:  # Image analysis
            if pred["is_deepfake"]:
                explanation = f"This image appears to have been manipulated using deepfake technology, which means it might have been created or altered by AI. The tool is {confidence:.2f} confident of this result, meaning there's a strong chance this image isn't real."
            else:
                explanation = f"This image appears to be genuine, with no signs of deepfake manipulation. The tool is {confidence:.2f} confident of this result, meaning it's very likely this image hasn't been altered by AI techniques."
        html_content += f"<p class='explanation'>{explanation}</p>"

        if "frame_details" in pred:  # For video
            html_content += "<h4>Frame-by-Frame Analysis (Sampled):</h4><table><tr><th>Frame No.</th><th>Status</th><th>Confidence</th></tr>"
            for fd in pred["frame_details"]:
                f_status = "Deepfake" if fd["is_deepfake"] else "Authentic"
                f_status_class = "status-deepfake" if fd["is_deepfake"] else "status-authentic"
                frame_confidence = (1.0 - fd["deepfake_score"]) if fd["is_deepfake"] else fd["deepfake_score"]
                html_content += f"<tr><td>{fd['frame_number']}</td><td><span class='{f_status_class}'>{f_status}</span></td><td>{frame_confidence:.2f}</td></tr>"
            html_content += "</table>"
        if "heatmap_base64" in pred and pred["heatmap_base64"]:
             html_content += f'<h4>Manipulation Heatmap (if available)</h4><img src="data:image/png;base64,{pred["heatmap_base64"]}" alt="Heatmap" class="report-image" style="max-width:300px;">'
        html_content += "</div>"

    if "copy_move_results" in results:
        cm_results = results["copy_move_results"]
        status_class = "status-tampered" if cm_results["tampering_detected"] else "status-not-tampered"
        status_text = "Tampered (Copy-Move Detected)" if cm_results["tampering_detected"] else "No Copy-Move Tampering Detected"
        html_content += f"<div class='section'><h3>Copy-Move Forgery Detection</h3>"
        html_content += f"<p>Status: <span class='{status_class}'>{status_text}</span></p>"
        html_content += f"<p>Details: {cm_results['details']}</p>"

        # Dynamic explanation for Copy-Move Forgery Detection
        if cm_results["tampering_detected"]:
            num_duplicates = int(cm_results["details"].split()[1])  # Extract number of duplicated regions
            explanation = f"This image shows signs of tampering because {num_duplicates} area(s) were found to be copied and pasted within the same image. This kind of tampering is often used to hide something or create a false impression, like duplicating an object or covering up a part of the image."
        else:
            explanation = "This image doesn't show any signs of copy-move tampering, which means no parts of the image were copied and pasted within itself. This suggests that this specific type of editing hasn't been done."
        html_content += f"<p class='explanation'>{explanation}</p>"

        if cm_results["marked_image_bytes"]:
            marked_image_base64 = base64.b64encode(cm_results["marked_image_bytes"]).decode('utf-8')
            html_content += f'<img src="data:image/png;base64,{marked_image_base64}" alt="Marked Image" class="report-image" style="max-width:500px;">'
        html_content += "</div>"

    if "statistical_analysis" in results:
        stat_results = results["statistical_analysis"]
        status_class = "status-anomaly" if stat_results["overall_anomaly_detected"] else "status-no-anomaly"
        status_text = "Anomaly Detected" if stat_results["overall_anomaly_detected"] else "No Anomaly Detected"
        html_content += f"<div class='section'><h3>Statistical Analysis</h3>"
        html_content += f"<p>Status: <span class='{status_class}'>{status_text}</span></p>"
        
        if "error" not in stat_results:
            html_content += "<h4>Color Histogram Analysis</h4>"
            html_content += f"<p>Variance - R: {stat_results['histogram_variance']['R']:.2f}, G: {stat_results['histogram_variance']['G']:.2f}, B: {stat_results['histogram_variance']['B']:.2f}</p>"
            html_content += f"<p>Histogram Anomaly: {'Yes' if stat_results['histogram_anomaly_detected'] else 'No'}</p>"
            
            html_content += "<h4>DCT Coefficient Analysis</h4>"
            html_content += f"<p>High-Frequency Ratio: {stat_results['dct_high_freq_ratio']:.4f}</p>"
            html_content += f"<p>DCT Anomaly: {'Yes' if stat_results['dct_anomaly_detected'] else 'No'}</p>"
            
            # Dynamic explanation
            if stat_results["overall_anomaly_detected"]:
                explanation = "Statistical anomalies were detected in this image. "
                if stat_results["histogram_anomaly_detected"]:
                    explanation += "The color histograms show unusual variance, which can happen when parts of the image are edited or combined from different sources, leading to inconsistent color distributions. "
                if stat_results["dct_anomaly_detected"]:
                    explanation += "The DCT coefficients have a high proportion of high-frequency components, which might indicate tampering, as manipulated regions often introduce unnatural frequency patterns."
            else:
                explanation = "No statistical anomalies were detected. The color histograms and DCT coefficients appear consistent with an authentic image, suggesting no obvious signs of manipulation based on these features."
            html_content += f"<p class='explanation'>{explanation}</p>"
        else:
            html_content += f"<p>Error: {stat_results['error']}</p>"
        html_content += "</div>"

    if "blending_results" in results:
        blend_results = results["blending_results"]
        status_class = "status-tampered" if blend_results["blending_detected"] else "status-not-tampered"
        status_text = "Blending Detected" if blend_results["blending_detected"] else "No Blending Detected"
        html_content += f"<div class='section'><h3>Blending Analysis</h3>"
        html_content += f"<p>Status: <span class='{status_class}'>{status_text}</span></p>"
        html_content += f"<p>Details: {blend_results['details']}</p>"
        
        # Dynamic explanation
        if blend_results["blending_detected"]:
            explanation = f"Signs of blending were detected, with {blend_results['blending_ratio']:.4f} of the image showing sharp transitions and edge inconsistencies. This suggests that parts of the image may have been copied and pasted from another source, a common tampering technique."
        else:
            explanation = "No significant blending was detected. The image shows consistent edges and gradients, suggesting that it hasn't been obviously combined from multiple sources."
        html_content += f"<p class='explanation'>{explanation}</p>"
        
        if blend_results["marked_image_bytes"]:
            marked_image_base64 = base64.b64encode(blend_results["marked_image_bytes"]).decode('utf-8')
            html_content += f'<img src="data:image/png;base64,{marked_image_base64}" alt="Blended Regions" class="report-image" style="max-width:500px;">'
        html_content += "</div>"

    if "enhancement_results" in results:
        enhance_results = results["enhancement_results"]
        status_class = "status-enhanced" if (enhance_results["sharpening_detected"] or enhance_results["noise_reduction_detected"]) else "status-not-enhanced"
        status_text = "Enhancements Detected" if (enhance_results["sharpening_detected"] or enhance_results["noise_reduction_detected"]) else "No Enhancements Detected"
        html_content += f"<div class='section'><h3>Enhancement Detection</h3>"
        html_content += f"<p>Status: <span class='{status_class}'>{status_text}</span></p>"
        html_content += f"<p>Sharpening: {'Yes' if enhance_results['sharpening_detected'] else 'No'} (Laplacian Variance: {enhance_results['sharpening_laplacian_variance']:.2f})</p>"
        html_content += f"<p>Noise Reduction: {'Yes' if enhance_results['noise_reduction_detected'] else 'No'} (Low Variance Ratio: {enhance_results['noise_reduction_low_variance_ratio']:.4f})</p>"
        
        # Dynamic explanation
        if enhance_results["sharpening_detected"] or enhance_results["noise_reduction_detected"]:
            explanation = "Post-processing enhancements were detected. "
            if enhance_results["sharpening_detected"]:
                explanation += "The image shows signs of sharpening, indicated by a high Laplacian variance, which can exaggerate edges and potentially hide tampering evidence. "
            if enhance_results["noise_reduction_detected"]:
                explanation += "The image appears to have undergone noise reduction, with many regions showing low variance, which can smooth out details and obscure signs of manipulation."
        else:
            explanation = "No significant post-processing enhancements were detected. The image does not show signs of sharpening or noise reduction, suggesting that any tampering evidence is less likely to be obscured by such techniques."
        html_content += f"<p class='explanation'>{explanation}</p>"
        html_content += "</div>"

    if "ela_image_bytes" in results or ("ela_performed" in results and results["ela_performed"]):
        html_content += "<div class='section'><h3>Error Level Analysis (ELA)</h3>"
        if "ela_image_bytes" in results:
            ela_base64 = base64.b64encode(results["ela_image_bytes"]).decode('utf-8')
            html_content += f'<img src="data:image/png;base64,{ela_base64}" alt="ELA Image" class="report-image" style="max-width:500px;">'
        else:
            html_content += "<p>ELA was performed (image not embedded).</p>"

        # Dynamic explanation for ELA
        explanation = "This image shows the result of Error Level Analysis, which looks for differences in how the image was compressed. If you see bright or uneven patches, it might mean parts of the image were edited or tampered with, because edited areas often compress differently. If the image looks mostly uniform, it's more likely to be authentic."
        if "ela_variance" in results:
            ela_variance = results["ela_variance"]
            if ela_variance > 500:  # Example threshold
                explanation += " The analysis shows a high level of variation in compression, which often indicates tampering."
            else:
                explanation += " The compression appears mostly uniform, which is typical for an authentic image."
        html_content += f"<p class='explanation'>{explanation}</p>"
        html_content += "</div>"

    if "metadata" in results:
        metadata = results["metadata"]
        html_content += "<div class='section'><h3>Metadata</h3><pre>"
        html_content += json.dumps(metadata, indent=2, ensure_ascii=False)
        html_content += "</pre>"

        # Dynamic explanation for Metadata
        has_editing_software = any(
            "photoshop" in str(value).lower() or "gimp" in str(value).lower() or "paint" in str(value).lower()
            for value in metadata.values()
        )
        has_errors = "ExifTool Status" in metadata or "ExifTool Execution Error" in metadata
        creation_date = metadata.get("EXIF:DateTimeOriginal") or metadata.get("File:FileCreateDate")

        if has_errors:
            explanation = "The metadata for this image couldn't be fully retrieved due to technical issues. Metadata is like a digital footprint that can tell us how the image was created or edited. Without it, we can't say much about the image's history, but this doesn't necessarily mean the image is tampered with."
        elif has_editing_software:
            explanation = "The metadata shows signs that this image might have been edited with software like Photoshop or GIMP. This doesn't prove the image is fake, but it suggests it may have been altered after it was originally created."
        elif creation_date:
            explanation = f"The metadata indicates this image was created on {creation_date}. There's no obvious sign of editing software, which means the image might be authentic, but metadata alone can't confirm this completely."
        else:
            explanation = "The metadata provides some basic information about the image, like its size and format, but there aren't enough details to determine if it was edited. This doesn't confirm tampering, but we can't rule it out either."
        html_content += f"<p class='explanation'>{explanation}</p>"
        html_content += "</div>"

    if "noise_analysis" in results and "estimated_noise_sigma" in results["noise_analysis"]:
        noise_val = results['noise_analysis']['estimated_noise_sigma']
        html_content += "<div class='section'><h3>Noise Analysis</h3>"
        html_content += f"<p>Estimated Noise Sigma: {noise_val if isinstance(noise_val, str) else f'{noise_val:.4f}'}</p>"

        # Dynamic explanation for Noise Analysis
        if isinstance(noise_val, str):
            explanation = "The tool couldn't analyze the noise patterns in this image due to technical issues. Noise patterns can help us spot tampering because edited areas often have different noise levels. Without this information, we can't use noise to assess the image's authenticity."
        else:
            if noise_val < 0.02:  # Typical threshold for low noise in authentic images
                explanation = f"The noise level in this image is very low (sigma = {noise_val:.4f}). This is what we'd expect from an authentic image taken by a camera, as real photos usually have consistent noise patterns. This suggests the image might be genuine."
            elif noise_val > 0.05:  # High noise might indicate tampering
                explanation = f"The noise level in this image is quite high (sigma = {noise_val:.4f}). This could be a sign of tampering, because edited areas often introduce inconsistent noise patterns. However, high noise can also occur in low-quality or heavily compressed images, so this isn't definitive proof of manipulation."
            else:
                explanation = f"The noise level in this image is moderate (sigma = {noise_val:.4f}). This doesn't strongly indicate tampering, but it also doesn't confirm the image is authentic. Noise patterns can vary depending on the image quality and editing, so this result is inconclusive."
        html_content += f"<p class='explanation'>{explanation}</p>"
        html_content += "</div>"
    
    html_content += "</body></html>"
    return html_content

# --- Streamlit Application ---
def main():
    st.set_page_config(layout="wide", page_title="Deepfake Forensics Tool")
    st.title("🕵️ Deepfake Detection and Forensic Investigation Tool")

    st.sidebar.title("About")
    st.sidebar.info(
        "This tool performs deepfake detection and forensic analysis on images and videos. "
        "It integrates various techniques to provide insights into media authenticity.\n\n"
        "SIT703 - Computer Forensics and Investigation\n"
        "S224784809 - SHREYAS NEERARAMBHAM MURALIKRISHNA"
    )
    if not TF_AVAILABLE:
        st.sidebar.error("TensorFlow is not available. Deepfake detection will not be available.")
    elif deepfake_model_loaded is None and MODEL_PATH:
        st.sidebar.error(f"TensorFlow model at {MODEL_PATH} could not be loaded.")
    elif deepfake_model_loaded is None and not MODEL_PATH:
        st.sidebar.error("Deepfake model path not set.")

    if not EXIFTOOL_AVAILABLE:
        st.sidebar.warning("pyexiftool library not found. Metadata analysis will be limited.")
    elif EXIFTOOL_EXECUTABLE and not os.path.exists(EXIFTOOL_EXECUTABLE):
        st.sidebar.warning(f"Custom ExifTool path set to '{EXIFTOOL_EXECUTABLE}' but not found. Will try system PATH.")
    elif not EXIFTOOL_EXECUTABLE:
         st.sidebar.info("EXIFTOOL_EXECUTABLE path not set. pyexiftool will try to find 'exiftool' in system PATH.")

    if not SKIMAGE_AVAILABLE:
        st.sidebar.warning("Scikit-image not found. Noise analysis will be limited.")

    uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "webp"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        file_type = uploaded_file.type.split('/')[0] 

        st.subheader(f"Uploaded: {file_name} ({file_type.capitalize()})")
        
        forensic_results = {
            "filename": file_name,
            "file_type": file_type
        }

        # Generate and store the hex dump for the entire file
        with st.spinner("Generating hex dump..."):
            forensic_results["hex_dump_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            hex_dump = generate_hex_dump(file_bytes)
            forensic_results["hex_dump"] = hex_dump

        if file_type == "image":
            try:
                pil_image = Image.open(BytesIO(file_bytes))
                st.image(pil_image, caption="Uploaded Image", use_container_width=True)
                image_array = np.array(pil_image.convert('RGB'))
            except Exception as e:
                st.error(f"Error opening image: {e}")
                return

            st.markdown("---")
            st.header("🔬 Image Analysis Results")

            # Updated tabs list without Reverse Image Search
            analysis_tabs = st.tabs([
                "Deepfake Detection",
                "Copy-Move Forgery Detection",
                "Statistical Analysis",
                "Blending Analysis",
                "Enhancement Detection",
                "Metadata",
                "Error Level Analysis (ELA)",
                "Noise Analysis",
                "Hex Dump"
            ])

            with analysis_tabs[0]:
                st.subheader("Deepfake Detection")
                with st.spinner("Analyzing for deepfake signatures..."):
                    try:
                        forensic_results["deepfake_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        current_pil_image_df = Image.open(BytesIO(file_bytes))
                        is_deepfake, score, heatmap_array = predict_deepfake_tf(current_pil_image_df.copy())
                        confidence = (1.0 - score) if is_deepfake else score
                        label = "Deepfake Confidence" if is_deepfake else "Confidence"
                        forensic_results["deepfake_prediction"] = {
                            "is_deepfake": is_deepfake,
                            "score": score,
                            "confidence_displayed": confidence
                        }
                        if is_deepfake:
                            st.error(f"Potential Deepfake Detected ({label}: {confidence:.2f})")
                        else:
                            st.success(f"Authentic ({label}: {confidence:.2f})")
                        if heatmap_array is not None:
                            st.info("Heatmap display/generation is a placeholder for actual implementation.")
                    except ValueError as e:
                        st.error(f"Deepfake detection failed: {e}")

            with analysis_tabs[1]:
                st.subheader("Copy-Move Forgery Detection")
                with st.spinner("Analyzing for copy-move forgery..."):
                    forensic_results["copy_move_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    cm_results = detect_copy_move_forgery(pil_image.copy())
                    forensic_results["copy_move_results"] = cm_results
                    if cm_results["tampering_detected"]:
                        st.error(f"Tampering Detected: {cm_results['details']}")
                        if cm_results["marked_image_bytes"]:
                            marked_image_pil = Image.open(BytesIO(cm_results["marked_image_bytes"]))
                            st.image(marked_image_pil, caption="Duplicated Regions Highlighted (Red Boxes, Green Lines)", use_container_width=True)
                    else:
                        st.success(f"No Copy-Move Tampering Detected: {cm_results['details']}")

            with analysis_tabs[2]:
                st.subheader("Statistical Analysis")
                with st.spinner("Performing statistical analysis..."):
                    forensic_results["statistical_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    stat_results = perform_statistical_analysis(pil_image.copy())
                    forensic_results["statistical_analysis"] = stat_results
                    
                    if "error" not in stat_results:
                        st.write("**Color Histogram Analysis**")
                        st.write(f"Variance - R: {stat_results['histogram_variance']['R']:.2f}, G: {stat_results['histogram_variance']['G']:.2f}, B: {stat_results['histogram_variance']['B']:.2f}")
                        st.write(f"Histogram Anomaly: {'Yes' if stat_results['histogram_anomaly_detected'] else 'No'}")
                        
                        st.write("**DCT Coefficient Analysis**")
                        st.write(f"High-Frequency Ratio: {stat_results['dct_high_freq_ratio']:.4f}")
                        st.write(f"DCT Anomaly: {'Yes' if stat_results['dct_anomaly_detected'] else 'No'}")
                        
                        if stat_results["overall_anomaly_detected"]:
                            st.error("Statistical Anomaly Detected")
                        else:
                            st.success("No Statistical Anomaly Detected")
                    else:
                        st.error(f"Analysis failed: {stat_results['error']}")

            with analysis_tabs[3]:
                st.subheader("Blending Analysis")
                with st.spinner("Analyzing for blended regions..."):
                    forensic_results["blending_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    blend_results = detect_blending(pil_image.copy())
                    forensic_results["blending_results"] = blend_results
                    
                    if blend_results["blending_detected"]:
                        st.error(f"Blending Detected: {blend_results['details']}")
                        if blend_results["marked_image_bytes"]:
                            marked_image_pil = Image.open(BytesIO(blend_results["marked_image_bytes"]))
                            st.image(marked_image_pil, caption="Blended Regions Highlighted (Green Overlay)", use_container_width=True)
                    else:
                        st.success(f"No Blending Detected: {blend_results['details']}")

            with analysis_tabs[4]:
                st.subheader("Enhancement Detection")
                with st.spinner("Checking for post-processing enhancements..."):
                    forensic_results["enhancement_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    enhance_results = detect_enhancements(pil_image.copy())
                    forensic_results["enhancement_results"] = enhance_results
                    
                    st.write(f"Sharpening: {'Yes' if enhance_results['sharpening_detected'] else 'No'} (Laplacian Variance: {enhance_results['sharpening_laplacian_variance']:.2f})")
                    st.write(f"Noise Reduction: {'Yes' if enhance_results['noise_reduction_detected'] else 'No'} (Low Variance Ratio: {enhance_results['noise_reduction_low_variance_ratio']:.4f})")
                    
                    if enhance_results["sharpening_detected"] or enhance_results["noise_reduction_detected"]:
                        st.warning("Post-Processing Enhancements Detected")
                    else:
                        st.success("No Post-Processing Enhancements Detected")

            with analysis_tabs[5]:
                st.subheader("Metadata Analysis")
                with st.spinner("Extracting metadata..."):
                    forensic_results["metadata_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    metadata = extract_metadata_with_exiftool(file_bytes, file_name)
                    forensic_results["metadata"] = metadata
                    st.text_area("Full Metadata (scrollable)", json.dumps(metadata, indent=2, ensure_ascii=False), height=300)

            with analysis_tabs[6]:
                st.subheader("Error Level Analysis (ELA)")
                with st.spinner("Performing ELA..."):
                    forensic_results["ela_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    ela_image_pil = Image.open(BytesIO(file_bytes))
                    ela_result_image, ela_variance = perform_ela(ela_image_pil.copy())
                    st.image(ela_result_image, caption="ELA Result", use_container_width=True)
                    
                    ela_bytes_io = BytesIO()
                    ela_result_image.save(ela_bytes_io, format="PNG")
                    forensic_results["ela_image_bytes"] = ela_bytes_io.getvalue()
                    forensic_results["ela_variance"] = ela_variance

            with analysis_tabs[7]:
                st.subheader("Noise Pattern Analysis")
                with st.spinner("Analyzing noise patterns..."):
                    forensic_results["noise_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    noise_data = analyze_noise_basic(image_array.copy())
                    forensic_results["noise_analysis"] = noise_data
                    if isinstance(noise_data.get("estimated_noise_sigma"), float):
                        st.write(f"Estimated Noise Sigma: {noise_data['estimated_noise_sigma']:.4f}")
                    else:
                        st.write(f"Estimated Noise Sigma: {noise_data.get('estimated_noise_sigma', 'N/A')}")

            with analysis_tabs[8]:
                st.subheader("Hex Dump")
                st.text_area("Hex Dump of the File (Entire File)", forensic_results["hex_dump"], height=300)
        
        elif file_type == "video":
            st.video(file_bytes)
            st.markdown("---")
            st.header("🔬 Video Analysis Results (Sampled Frames)")

            # Add a tab for Hex Dump in video analysis
            analysis_tabs = st.tabs(["Deepfake Detection", "Metadata", "Hex Dump"])

            with analysis_tabs[0]:
                temp_video_path = None
                try:
                    os.makedirs("uploads", exist_ok=True)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1], dir="uploads") as tmp_vid:
                        tmp_vid.write(file_bytes)
                        temp_video_path = tmp_vid.name

                    cap = cv2.VideoCapture(temp_video_path)
                    if not cap.isOpened():
                        st.error("Error: Could not open video file.")
                        return

                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps == 0: fps = 25
                    
                    st.write(f"Video details: {frame_count} frames, {fps:.2f} FPS")
                    max_frames_to_analyze = st.slider("Max frames to analyze from video:", 1, 20, 5)
                    frame_interval = int(fps * 2)
                    if frame_interval == 0: frame_interval = 1

                    analyzed_frames_count = 0
                    deepfake_scores = []
                    frame_results_list = []

                    st.subheader("Frame-by-Frame Deepfake Detection (Sampled)")
                    frame_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    forensic_results["deepfake_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

                    for i in range(frame_count):
                        ret, frame = cap.read()
                        if not ret: break
                        
                        if i % frame_interval == 0 and analyzed_frames_count < max_frames_to_analyze:
                            analyzed_frames_count += 1
                            status_text.info(f"Analyzing frame {i} ({analyzed_frames_count}/{max_frames_to_analyze})...")
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_frame = Image.fromarray(rgb_frame)
                            
                            with frame_placeholder.container():
                                st.image(pil_frame, caption=f"Analyzing Frame {i}", width=300)

                            try:
                                is_deepfake, score, _ = predict_deepfake_tf(pil_frame.copy())
                                confidence = (1.0 - score) if is_deepfake else score
                                label = "Deepfake Confidence" if is_deepfake else "Confidence"
                                deepfake_scores.append(score)
                                frame_results_list.append({"frame_number": i, "is_deepfake": is_deepfake, "deepfake_score": score})
                                status_text.info(f"Frame {i}: {'Potential Deepfake' if is_deepfake else 'Authentic'} ({label}: {confidence:.2f})")
                            except ValueError as e:
                                status_text.error(f"Frame {i} analysis failed: {e}")
                            time.sleep(0.1)

                        if analyzed_frames_count >= max_frames_to_analyze: break
                        progress_bar.progress(min(1.0, (analyzed_frames_count / max_frames_to_analyze if max_frames_to_analyze > 0 else 1.0)))

                    cap.release()
                    frame_placeholder.empty()
                    status_text.empty()
                    progress_bar.empty()

                    if deepfake_scores:
                        avg_deepfake_score = np.mean(deepfake_scores)
                        avg_confidence = (1.0 - avg_deepfake_score) if overall_video_deepfake else avg_deepfake_score
                        label = "Avg. Deepfake Confidence" if overall_video_deepfake else "Avg. Confidence"
                        overall_video_deepfake = avg_deepfake_score > DEEPFAKE_THRESHOLD
                        overall_video_deepfake = not overall_video_deepfake  # Invert based on class indices
                        st.subheader("Overall Video Assessment")
                        if overall_video_deepfake:
                            st.error(f"Video flagged as Potential Deepfake ({label}: {avg_confidence:.2f})")
                        else:
                            st.success(f"Video flagged as Authentic ({label}: {avg_confidence:.2f})")
                        forensic_results["deepfake_prediction"] = {
                            "is_deepfake": overall_video_deepfake, 
                            "score": avg_deepfake_score,
                            "confidence_displayed": avg_confidence,
                            "frame_details": frame_results_list
                        }
                    else:
                        st.write("No frames analyzed for deepfake detection, or video too short/low FPS.")

                except Exception as e:
                    st.error(f"Error processing video: {e}")
                finally:
                    if temp_video_path and os.path.exists(temp_video_path):
                        try: os.remove(temp_video_path)
                        except Exception as e_rem:
                             print(f"Error removing temp video file {temp_video_path}: {e_rem}")
                             st.warning(f"Could not remove temporary video file: {temp_video_path}")

            with analysis_tabs[1]:
                st.subheader("Video Metadata")
                with st.spinner("Extracting video metadata..."):
                    forensic_results["metadata_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    video_metadata = extract_metadata_with_exiftool(file_bytes, file_name)
                    forensic_results["metadata"] = video_metadata
                    st.text_area("Full Video Metadata (scrollable)", json.dumps(video_metadata, indent=2, ensure_ascii=False), height=300)

            with analysis_tabs[2]:
                st.subheader("Hex Dump")
                st.text_area("Hex Dump of the File (Entire File)", forensic_results["hex_dump"], height=300)

        st.markdown("---")
        st.header("📋 Forensic Report")
        if st.button("Generate Full Report"):
            with st.spinner("Generating report..."):
                time.sleep(1)
                html_report = generate_report_html(forensic_results)
                b64_html = base64.b64encode(html_report.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64_html}" download="forensic_report_{os.path.splitext(file_name)[0]}.html">Download HTML Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                with st.expander("Preview Report (HTML)"):
                    st.components.v1.html(html_report, height=600, scrolling=True)
    else:
        st.info("👋 Welcome! Please upload an image or video file to begin analysis.")

if __name__ == "__main__":
    main()