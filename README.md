# Feature Fusion

Feature Fusion is a Streamlit-based application that enables users to explore and visualize feature detection algorithms in computer vision. It supports **SIFT**, **SURF**, and **Harris Corner Detection** algorithms to detect and match key points or identify corners in images.

## Features
- **SIFT (Scale-Invariant Feature Transform):**
  - Detect and match key points between two images.
  - Visualize robust matches between images under varying conditions.
  
- **SURF (Speeded-Up Robust Features):**
  - Detect and match key points between two images with different scales and orientations.
  - Fast and efficient feature extraction.

- **Harris Corner Detection:**
  - Detect corners in a single grayscale image.
  - Visualize detected corners overlaid on the original image.

## Requirements
- Python 3.7 or later
- Streamlit
- OpenCV
- NumPy
- Pillow

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Feature-Fusion.git
   cd Feature-Fusion
   ```

2. Create a virtual environment (optional):
   ```bash
   python -m venv env
   source env/bin/activate 
   # On Windows use: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure you have OpenCV with Contrib modules installed for SURF support:
   ```bash
   pip install opencv-contrib-python
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Upload images and select an algorithm to visualize features:
   - For **SIFT** and **SURF**, upload two images to detect and match keypoints.
   - For **Harris Corner Detection**, upload a single image to visualize detected corners.

3. View the results displayed in the app interface.

## Instructions
- Upload images in JPEG or PNG format.
- Select an algorithm (`SIFT`, `SURF`, `Harris Corner`) from the options provided.
- Click the "Detect Features" button to visualize the results.

## Acknowledgments
- OpenCV library for feature detection and computer vision functions.
- Streamlit framework for creating interactive applications.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.