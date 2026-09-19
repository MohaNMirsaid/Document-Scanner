# Document Scanner & Face Detection

A Python-based document scanning tool that automatically detects the boundaries of a document, performs perspective correction, detects faces, extracts face regions, and converts the document into a high-contrast scanned image.

## Features

* Automatic document boundary detection
* Edge detection using Canny
* Contour detection for identifying the document
* Perspective transformation (four-point transform)
* Face detection using OpenCV Haar Cascade
* Automatic face cropping with padding
* Face image upscaling
* Adaptive thresholding for document scanning
* Automatic timestamp-based output filenames
* Preview of intermediate processing steps

## How It Works

The application processes an input image through several stages:

### 1. Image Loading and Resizing

The input image is loaded using OpenCV. A resized copy is used for faster processing while the original image is preserved for the final perspective transformation.

### 2. Edge Detection

The image is converted to grayscale and blurred using Gaussian Blur. Canny edge detection is then applied to identify the edges of objects in the image.

### 3. Document Contour Detection

The application finds contours in the edge image and examines the largest contours. It searches for a contour with four points, which is assumed to represent the document.

### 4. Perspective Transformation

After detecting the four corners of the document, the application performs a four-point perspective transformation.

This converts a document photographed from an angle into a flat, rectangular image similar to a scanned document.

### 5. Face Detection

The perspective-corrected document is passed to OpenCV's Haar Cascade face detector.

Detected faces are cropped with additional padding around them and saved as separate PNG files.

### 6. Adaptive Thresholding

The corrected document is converted to grayscale and adaptive Gaussian thresholding is applied.

This produces a high-contrast black-and-white version of the document.

### 7. Saving the Result

The processed document is automatically saved using a timestamp-based filename:

```text
scanned_image_YYYY-MM-DD_HH-MM-SS.png
```

Detected faces are saved using filenames such as:

```text
face_YYYY-MM-DD_HH-MM-SS_1.png
```

## Requirements

* Python 3.x
* OpenCV
* NumPy
* imutils
* scikit-image

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/document-scanner.git
```

Move into the project directory:

```bash
cd document-scanner
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the application from the command line:

```bash
python scannerApp.py --image path/to/your/image.jpg
```

For example:

```bash
python scannerApp.py --image images/id_card.jpg
```

The program will display intermediate processing stages and save the final scanned image automatically.

## Command-Line Arguments

| Argument  | Description             |
| --------- | ----------------------- |
| `-i`      | Path to the input image |
| `--image` | Path to the input image |

Example:

```bash
python scannerApp.py -i images/document.jpg
```

## Output

The program generates two types of output:

### Scanned Document

```text
scanned_image_2026-09-19_18-30-00.png
```

### Detected Faces

```text
face_2026-09-19_18-30-00_1.png
face_2026-09-19_18-30-00_2.png
```

## Technologies

* Python
* OpenCV
* NumPy
* scikit-image
* imutils
* Computer Vision
* Image Processing

## Project Pipeline

```text
Input Image
     │
     ▼
Resize Image
     │
     ▼
Grayscale + Gaussian Blur
     │
     ▼
Canny Edge Detection
     │
     ▼
Contour Detection
     │
     ▼
Find Four-Point Document Contour
     │
     ▼
Perspective Transformation
     │
     ├──────────────► Face Detection
     │                     │
     │                     ▼
     │                Face Cropping
     │                     │
     │                     ▼
     │                Save Face Images
     │
     ▼
Grayscale Conversion
     │
     ▼
Adaptive Thresholding
     │
     ▼
Scanned Document
     │
     ▼
Save Result
```

## Important Notes

The document detection works best when:

* The document has clearly visible boundaries.
* The four corners of the document are visible.
* There is sufficient contrast between the document and its background.
* The input image is reasonably well lit.
* The document is not heavily occluded.

The face detector also depends on the quality, size, and orientation of the detected face.

## Future Improvements

Possible improvements for future versions include:

* Adding a graphical user interface
* Supporting multiple input images
* Automatic image rotation
* Better document boundary detection
* Improved face detection using modern deep-learning models
* Automatic document enhancement
* Shadow and background removal
* OCR for extracting text from scanned documents
* Saving results to a dedicated output directory
* Adding configuration options for thresholding parameters

## License

This project is intended for educational and computer vision development purposes.
