# Women Safety Surveillance System
**Real-Time Threat Detection with Web Interface & Alerts**

[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-blue)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0-red)](https://ultralytics.com/)

## 🎯 AIM
Enhance **women's safety** through AI-powered surveillance that detects lone women in public spaces during nighttime, triggering real-time browser alerts and SMS notifications for immediate intervention.

## 🌟 Key Features
- **Real-Time Video Analytics Dashboard**
- **Automatic Night Detection** (6 PM - 6 AM)
- **Visual Alert** for lone women detection
- **JSON API Endpoint** for system data
- **Multi-Platform Support**: Webcams, IP cameras, video files
- **Cloud Deployment Ready**

## 🚀 Recent Enhancements
1. **Web Interface Integration**
   - Flask-based dashboard with live video streaming
   - Real-time statistics overlay
   - Browser push notifications
2. **Alert System Upgrades**
   - Visual popup alerts with acknowledgment
3. **Deployment Flexibility**
   - Local hosting with ngrok support
   - Cloud deployment guides (Render, PythonAnywhere)

## 🛠️ Technical Stack
| Component              | Technology          |
|------------------------|---------------------|
| Object Detection       | YOLOv8 Nano         |
| Gender Analysis         | DeepFace            |
| Object Tracking         | Custom CentroidTracker |
| Web Framework           | Flask               |
| Data Persistence        | JSON + Timestamp    |

## 🖥️ Web Interface Features
- Live video stream with annotations
- Real-time counters (People/Men/Women)
- Lone women detection indicator


## 🚦 Getting Started

### Prerequisites
```bash
# Core Dependencies
pip install -r requirements.txt

# Optional for GPU Support
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117


# Team Details
- **Team Name:** Syntax Error
- **Members:**
 1. Aayush Kumar Singh - AI/ML, Computer vision & Web App
 2. Adesh Dutta
 3. Aman Singh
