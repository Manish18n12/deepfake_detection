Real vs Fake Face Detection (Deepfake Detection
System - 2026)
An AI-powered web application that detects whether a face image is Real or Fake using Deep
Learning. This project uses a Convolutional Neural Network (CNN) trained on the Kaggle 'Real and
Fake Face Detection' dataset and provides a Flask-based web interface for predictions.
Dataset
Dataset Name: Real and Fake Face Detection
Kaggle Link: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection
Note: The dataset is NOT included in this repository due to size and licensing restrictions.
Project Structure
Deepfake-Detection-Project/
 app.py
 train_model.py
 predict.py
 prepare_data.py
 requirements.txt
 README.md
 templates/
 index.html
 gallery.html
 static/
 style.css
 uploads/
 dataset/ (NOT included)
 deepfake_model.h5 (optional if <100MB)
 .gitignore
Installation & Setup
1. Clone the repository:
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
2. Create virtual environment:
   python -m venv venv
3. Activate it:
   Windows: venv\Scripts\activate
   macOS/Linux: source venv/bin/activate
4. Install dependencies:
   pip install -r requirements.txt
Model Training
python prepare_data.py
python train_model.py
The trained model will be saved as:
deepfake_model.h5
Run the Web Application
python app.py
Open in browser:
http://localhost:5000
Features
• Deep Learning CNN model
• Image preprocessing pipeline
• Flask web interface
• Real-time prediction
• Organized project structure
Author
Manish N
B.Tech AI Student
Deep Learning & AI Enthusiast
