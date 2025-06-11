from inference_sdk import InferenceHTTPClient
import cv2
import os
import glob

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="RAO3qcOxTrwgNcnSGrMD"
)

# Configuration
DATASET_PATH = "D:/Hand Gesture recognition/dataset"
MODEL_ID = "hand-gesture-recognition-y5827/2"
OUTPUT_DIR = os.path.join(DATASET_PATH, "output")
IMAGE_DIR = os.path.join(DATASET_PATH, "images")

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def infer_image(image_path, output_path):
    """Perform inference on a single image using Roboflow model."""
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return False
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return False
    
    # Perform inference
    try:
        result = CLIENT.infer(image_path, model_id=MODEL_ID)
    except Exception as e:
        print(f"Error during inference for {image_path}: {e}")
        return False
    
    # Process results
    if 'predictions' not in result:
        print(f"No predictions found for {image_path}")
        return False
    
    for prediction in result['predictions']:
        # Extract prediction details
        x = int(prediction['x'])
        y = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])
        class_name = prediction['class']
        confidence = prediction['confidence']
        
        print(f"{os.path.basename(image_path)} - Detected: {class_name} with confidence {confidence:.2f}")
        
        # Calculate bounding box coordinates
        x1 = x - width // 2
        y1 = y - height // 2
        x2 = x + width // 2
        y2 = y + height // 2
        
        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{class_name} {confidence:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save result
    cv2.imwrite(output_path, img)
    print(f"Output saved at: {output_path}")
    return True

def main():
    """Main function to run inference on all images in the dataset."""
    ensure_output_dir()
    
    # Get all image files
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return
    
    # Process each image
    for image_path in image_files:
        # Generate unique output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_output.jpg")
        infer_image(image_path, output_path)
    
    # Display all output images
    output_images = glob.glob(os.path.join(OUTPUT_DIR, "*.jpg"))
    for output_image in output_images:
        img = cv2.imread(output_image)
        if img is not None:
            cv2.imshow('Result', img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()