# Hand-Gesture-Recognition

This project uses a Roboflow-hosted machine learning model to perform hand gesture recognition on images. The script processes images from a specified dataset directory, performs inference using the Roboflow Inference API, draws bounding boxes around detected gestures, and saves the annotated images to an output directory. The results are also displayed for visual verification.

Prerequisites





Python 3.7 or higher



A Roboflow account with an API key



A dataset of images (in .jpg or .png format) for gesture recognition



Internet connection for API calls to Roboflow

Installation





Clone the Repository (if applicable):

git clone <repository-url>
cd hand-gesture-recognition



Install Dependencies: Create a virtual environment (optional but recommended) and install the required packages:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt



Set Up the Dataset:





Place your dataset images (.jpg or .png) in the dataset/images directory.



Ensure the directory structure is as follows:

dataset/
└── images/
    ├── image1.jpg
    ├── image2.png
    └── ...



Configure the Script:





Open main.py and update the following variables:





DATASET_PATH: Path to your dataset directory (e.g., D:/Hand Gesture recognition/dataset).



MODEL_ID: Your Roboflow model ID (e.g., hand-gesture-recognition-y5827/2).



api_key: Your Roboflow API key in the CLIENT initialization.

Usage





Run the Script: Execute the main script to process all images in the dataset:

python main.py



Output:





The script will:





Create an output directory inside DATASET_PATH if it doesn't exist.



Process each image in the images directory, perform inference, and draw bounding boxes with gesture labels and confidence scores.



Save annotated images to the output directory with _output.jpg appended to the filename.



Display each annotated image in a window (press any key to move to the next image).



Console output will show detected gestures, confidence scores, and file paths for saved images.



Example Output: For an image named hand_wave.jpg, the script might output:

hand_wave.jpg - Detected: Wave with confidence 0.92
Output saved at: dataset/output/hand_wave_output.jpg

Dependencies

See requirements.txt for the list of required Python packages.

Notes





API Key Security: Keep your Roboflow API key secure and do not share it publicly.



Image Formats: The script supports .jpg and .png images. Add support for other formats by modifying the glob patterns in main.py.



Error Handling: The script includes basic error handling for missing images, failed inference, or invalid API responses.



Display Window: The script uses OpenCV's imshow to display results. Ensure you have a graphical environment if running on a desktop system, or modify the script to skip the display step for headless systems.



Pyodide Compatibility: The current script uses file I/O and OpenCV's display functions, which are not compatible with Pyodide. For browser-based execution, significant modifications would be required (e.g., removing file I/O and using a canvas for display).
