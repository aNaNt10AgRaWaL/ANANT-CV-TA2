import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Feature Detection App", layout="centered")

st.title("Feature Fusion")

algo = st.radio("Choose Algorithm", ["SIFT", "SURF", "Harris Corner"])

def upload_image(label):
    uploaded_file = st.file_uploader(label, type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        return image
    return None

def sift_match(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    result = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    return result

def surf_match(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

    kp1, des1 = surf.detectAndCompute(gray1, None)
    kp2, des2 = surf.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    result = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    return result

def harris_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img

if algo == "SIFT":
    st.subheader("SIFT Keypoint Matching")
    img1 = upload_image("Upload First Image")
    img2 = upload_image("Upload Second Image")
    if img1 is not None and img2 is not None:
        result = sift_match(img1, img2)
        st.image(result, channels="RGB", caption="SIFT Matches")

elif algo == "SURF":
    st.subheader("SURF Keypoint Matching")
    img1 = upload_image("Upload First Image")
    img2 = upload_image("Upload Second Image")
    if img1 is not None and img2 is not None:
        result = surf_match(img1, img2)
        st.image(result, channels="RGB", caption="SURF Matches")

elif algo == "Harris Corner":
    st.subheader("Harris Corner Detection")
    img = upload_image("Upload Image")
    if img is not None:
        result = harris_corners(img.copy())
        st.image(result, channels="RGB", caption="Harris Corners")

st.write("""
### Instructions:
1. Upload an image for each algorithm.
2. For SIFT and SURF, upload two images to compare keypoints.
3. For Harris Corner, upload a single image to detect corners.
4. Choose an algorithm and visualize the results.
""")