# AI-Powered-Surveillance-System


An intelligent video surveillance system that automatically detects suspicious activities using deep learning and computer vision techniques. Designed to improve public safety in environments such as parking lots, banks, and campuses, this system leverages anomaly detection to monitor CCTV footage in real-time without requiring human intervention.

📌 Features

Loitering Detection – Identifies individuals staying too long in restricted areas.

Unusual Movement Patterns – Detects erratic or abnormal movements using optical flow analysis.

Object Abandonment – Recognizes unattended objects through background subtraction.

Crowd Anomalies – Identifies unusual crowd formations and behaviors.

Real-Time Alerts – Processes at 15–20 FPS with instant notifications.

Explainable AI – Provides visual cues on why an event was flagged as anomalous.

Synthetic Data Generation – Uses GANs to create rare-event data for improved training.

Interactive Dashboard – Streamlit-powered UI for live monitoring and visualization.

🏗️ System Architecture

The system follows a multi-stage pipeline:

Data Preparation – Preprocessing of the Avenue Dataset with spatial and temporal augmentations.

Object Detection & Tracking – YOLOv8 with Hungarian + Kalman filtering.

Anomaly Detection – Isolation Forest + LSTM for sequence modeling.

Synthetic Data Generation – GAN-based augmentation for rare anomalies.

System Integration – Streamlit dashboard + SQLite database for alerts and logs.

⚙️ Tech Stack

Languages & Frameworks

Python 3.8+

OpenCV 4.7.0

PyTorch 2.0.0

Ultralytics YOLOv8

Streamlit 1.22.0

Scikit-learn 1.2.2

Models & Algorithms

YOLOv8n (real-time object detection)

Isolation Forest (unsupervised anomaly detection)

LSTM Networks (behavioral sequence modeling)

Custom GAN (synthetic anomaly generation)

Optical Flow (Farneback method for motion estimation)

Other Libraries

SQLite3 (database for alerts)

NumPy, Pandas, Matplotlib, Plotly


🚀 Installation

Clone the repository:

git clone https://github.com/your-username/AI-Powered-Surveillance-System.git
cd AI-Powered-Surveillance-System


Install dependencies:

pip install -r requirements.txt


Download and prepare the Avenue Dataset


Run the main system:

python surveillance_main_system.py


Launch the dashboard:

streamlit run surveillance_dashboard.py

📊 Performance

Detection Accuracy (mAP): 89.2% (Avenue Dataset)

Processing Speed: 15–20 FPS on NVIDIA RTX 3060

False Positive Rate: <8%

🎥 Demo

📌 Demo Video
https://drive.google.com/file/d/15YT9GmwSb2yBl7YUQcdt3mz4Zg1ZBldw/view?usp=drive_link



📖 References

Avenue Dataset https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

YOLOv8 Documentation

OpenCV Docs

PyTorch Docs

Streamlit Docs

👩‍💻 Author

Gauri Pandey
AI & Machine Learning Enthusiast 


🛡️ NetraGuard — Intelligent AI Surveillance System

NetraGuard is an enterprise-grade AI-powered video surveillance platform designed for real-time anomaly detection, incident analytics, and intelligent monitoring of CCTV and surveillance footage.
It combines deep learning, computer vision, and data analytics with an interactive dashboard for security-focused use cases.

🚀 Key Highlights

🔍 Real-time anomaly detection from CCTV footage

🎯 Intelligent incident classification with confidence scoring

📊 Live analytics dashboard with timelines & distributions

🧠 YOLOv8-based object detection for high-speed inference

🗂️ SQLite-backed alert logging for persistence & auditing

📈 Exportable security reports (CSV / JSON / TXT)

🖥️ Modern cyber-themed UI built with Streamlit

🧠 System Capabilities
🔐 Anomaly Detection

NetraGuard detects abnormal activities such as:

Suspicious loitering

Unusual motion patterns

Unexpected object presence

Behavior deviations over time

📊 Security Analytics Dashboard

The dashboard provides:

Incident metrics (24h)

Severity-based alert prioritization

Confidence distribution analysis

Hourly & temporal alert timelines

Export-ready security reports

🗃️ Alert Persistence

All detected incidents are stored locally using SQLite, enabling:

Historical analysis

Audit trails

System reliability without cloud dependency

🏗️ Architecture Overview
Video Input
   ↓
YOLOv8 Object Detection
   ↓
Anomaly Scoring Logic
   ↓
Alert Generation
   ↓
SQLite Database
   ↓
Streamlit Dashboard (Analytics & Controls)

🧰 Tech Stack
Core Technologies

Python 3.10

OpenCV

Ultralytics YOLOv8

PyTorch

NumPy / Pandas

Dashboard & Visualization

Streamlit

Plotly

Custom CSS (Cyber UI Theme)

Data & Storage

SQLite3

📁 Project Structure
NetraGuard/
│
├── netraguard_main_system.py      # Core detection & processing engine
├── surveillance_dashboard.py      # Streamlit dashboard
├── surveillance_system.db         # SQLite alert database
├── yolov8n.pt                     # YOLOv8 model weights
├── requirements.txt
├── README.md
└── sample_video/
    └── demo.mp4

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/NetraGuard.git
cd NetraGuard

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Running the System
🔹 Run Core Detection (CLI)
python netraguard_main_system.py --input sample_video/demo.mp4

🔹 Launch Dashboard
streamlit run surveillance_dashboard.py


Open in browser:

http://localhost:8501

🎥 Demo Video

📌 Project Demo (Dashboard + Detection)
👉 Add your Google Drive / YouTube demo link here

The demo showcases video upload, anomaly detection, alert generation, analytics visualization, and export features.

🧪 Design Considerations

Built with defensive programming to handle missing or partial data

Designed for local execution (no cloud dependency)

UI optimized for security & monitoring use cases

Modular structure for future expansion (email alerts, RTSP streams, cloud DBs)

🛠️ Future Enhancements

Live CCTV / RTSP stream support

Email / webhook alerting

Role-based access control

Cloud deployment (AWS / Azure)

Model fine-tuning for domain-specific environments

👤 Author

Supriya Mishra
AI & Software Engineering Enthusiast
Focused on building intelligent, scalable, and real-world systems.

📄 License

This project is licensed under the MIT License — free to use, modify, and distribute.
