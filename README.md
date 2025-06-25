# DeepFake-Detection-and-Image-Forensics-Tool
This project implements a web-based application using Streamlit for detecting deepfakes and performing forensic analysis on images and videos. The tool integrates multiple techniques to assess the authenticity of digital media, including deepfake detection, copy-move forgery detection, statistical analysis, blending analysis, enhancement detection, metadata analysis, error level analysis (ELA), noise pattern analysis, and hex dump viewing.
Usage Instructions
Setup

Ensure Python is Installed:

Verify that Python 3.8 or higher is installed on your system. You can check this by running:python --version


If Python is not installed, download and install it from python.org.


Install Dependencies:

Copy the project files (app.py, mesonet_final.h5, requirements.txt, and this README) to a local directory on your machine.
Open a terminal or command prompt and navigate to the project directory:cd path/to/your/project/directory


Install the required Python packages using:pip install -r requirements.txt




Place the Model File:

Ensure the trained model file mesonet_final.h5 is in the same directory as app.py.


Install ExifTool:

The tool uses ExifTool for metadata extraction. Download ExifTool from the official website and install it.
After installation, ensure ExifTool is accessible via the system PATH, or update the EXIFTOOL_EXECUTABLE variable in app.py to point to the exiftool.exe file (e.g., D:\Forensics HD\exiftool.exe).



Running the Application
Run the Streamlit app from the project directory:
streamlit run app.py


This will launch the web application in your default browser. If it doesn’t open automatically, navigate to the URL displayed in the terminal (typically http://localhost:8501).

Using the App

Upload a File:

Upload an image (jpg, jpeg, png, webp) or video (mp4, mov, avi) file via the web interface.


Analyze the File:

For images, analysis tabs include Deepfake Detection, Copy-Move Forgery Detection, Statistical Analysis, Blending Analysis, Enhancement Detection, Metadata, ELA, Noise Analysis, and Hex Dump.
For videos, tabs include Deepfake Detection, Metadata, and Hex Dump.


View Results:

Each tab displays specific analysis results, such as confidence scores for deepfake detection or marked images for tampering.


Generate Report:

Click "Generate Full Report" to create an HTML report summarizing all analyses (excluding the hex dump).



Dependencies

Python 3.8+
streamlit
tensorflow==2.10.0
numpy
pillow
opencv-python
scikit-image
pyexiftool
scipy

For a complete list, see requirements.txt.
Additional Notes

The hex dump feature displays the entire file’s hexadecimal content but is not included in the generated report.
Ensure mesonet_final.h5 is in the same directory as app.py.
For video analysis, the tool samples frames at intervals to manage processing time.
If ExifTool is not in the system PATH, update the EXIFTOOL_EXECUTABLE variable in app.py with the correct path to the ExifTool executable.
