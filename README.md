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
