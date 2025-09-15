# 🎾 AI-Powered Tennis Game Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-CV2-green" alt="OpenCV">
  <img src="https://img.shields.io/badge/DeepLearning-PyTorch-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="status">
</p>

## 📌 Project Overview  

This project leverages **Computer Vision** and **Deep Learning** to perform a detailed **AI-powered tennis match analysis**.  
The model detects tennis balls in real-time, tracks their movement, and provides valuable insights into the game dynamics.  

By combining **object detection, tracking, and trajectory analysis**, this system can help coaches, analysts, and players better understand match performance.  

---

## 🎥 Example Output  

Here’s what the AI-powered tennis ball detection looks like in action:  

<p align="center">
  <img src="output_videos/example-gif.gif" alt="Tennis Ball Detection Example" width="600"/>
</p>  


---

## ⚡ Features  

- 🎾 **Tennis Ball Detection** using a custom-trained model.  
- 🏸 **Trajectory Tracking** of the ball across frames.  
- 📊 **Game Analysis**: speed estimation, shot distribution, and positional analysis.  
- 🤖 **Deep Learning Backbone** for robust detection under varying lighting and camera angles.  
- 📈 **Extendable Framework** – can be adapted for racket tracking, player movement, or advanced stats.  

---

## 📂 Dataset  

We used the **Tennis Ball Detection Dataset** from [Roboflow](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6).  

- ✅ Annotated tennis ball positions in frames.  
- ✅ Multiple scenarios and camera angles.  
- ✅ Preprocessed for training YOLO-based models.  

---

## 🧠 How the AI Works  

This project leverages **Deep Learning** and **Computer Vision** to detect and track tennis balls in real-time. At its core, it uses a **Convolutional Neural Network (CNN)** based object detection model (like YOLOv5/YOLOv8) combined with tracking algorithms for trajectory analysis.  

### 🔹 Data Flow Overview  

1. **Video Input**  
   - Raw tennis match footage is split into frames.  
   - Each frame is preprocessed: resized, normalized, and augmented to improve model robustness.  

2. **Feature Extraction (Convolutional Layers)**  
   - The CNN applies **convolutional filters** to extract spatial features: edges, corners, textures.  
   - Multiple layers allow the network to learn **hierarchical representations**:  
     - Early layers detect simple features (edges, color gradients).  
     - Mid layers detect shapes and textures (tennis ball contours).  
     - Deep layers detect higher-level concepts (ball in motion, occlusions).  

   ![Convolutional Layers](https://miro.medium.com/v2/resize:fit:1200/1*tCGkL3vJ3-zp9K1oCmFhIA.png)      
   *Illustration of convolutional layers extracting features from an input frame.*

3. **Detection Head**  
   - After feature extraction, the **detection head** predicts bounding boxes and confidence scores.  
   - YOLO divides the image into grids and predicts object locations and probabilities per grid cell.  

   ![YOLO Detection Head](https://user-images.githubusercontent.com/65339139/253833314-26fc3681-358d-4c09-a841-7db328fabf6c.png)  
   *How YOLO predicts bounding boxes and confidence scores from feature maps.*

4. **Non-Maximum Suppression (NMS)**  
   - Eliminates overlapping predictions to keep only the most confident bounding box per detected ball.  

5. **Tracking Across Frames**  
   - Uses algorithms like **SORT** or **DeepSORT** to link detected balls across consecutive frames.  
   - Ensures smooth trajectory even when the ball is temporarily occluded.  

   ![Tracking Diagram](https://miro.medium.com/v2/resize:fit:1400/0*-S2EkuGhkP9tp9It.JPG)  
   *Tracking module connecting ball positions across frames.*

6. **Trajectory and Speed Analysis**  
   - Extracts the path of the ball over time.  
   - Computes metrics such as speed, bounce points, and shot angles.  
   - This provides valuable game insights for coaches and players.  

### 🔹 Neural Network Architecture (Detailed)  

| Layer Type           | Purpose                                                                 |
|----------------------|-------------------------------------------------------------------------|
| Input Layer           | Receives raw frames (H×W×3).                                           |
| Convolution + ReLU    | Detects local patterns (edges, corners).                                |
| Max Pooling           | Reduces spatial dimensions while preserving important features.         |
| Residual/Backbone     | Deep layers (e.g., CSPDarknet in YOLOv5) extract higher-level features.|
| Detection Layer       | Outputs bounding boxes, class probabilities, and confidence scores.     |
| Post-processing (NMS) | Filters redundant boxes, keeps the best predictions.                    |

   ![Model Architecture](assets/model_architecture.png)  
   *Full CNN + detection architecture showing flow from input frame to output bounding boxes.*

This layered design allows the model to **accurately detect small, fast-moving objects** like tennis balls, even in challenging lighting or when partially occluded by players.  

---

---

## ⚙️ Requirements  

Make sure you have the following installed:  

- Python **3.9+**  
- [PyTorch](https://pytorch.org/)  
- OpenCV  
- NumPy  
- Matplotlib  
- Roboflow API (optional for dataset loading)  

Install dependencies via:  

```bash
pip install -r requirements.txt
```

---

## 🚀 Installation & Setup  

Clone the repository:  

```bash
git clone https://github.com/danielgavrila2/Tennis-Game-Analysis.git
cd Tennis-Game-Analysis
```

Install dependencies:  

```bash
pip install -r requirements.txt
```

(Optional) Download dataset directly from Roboflow:  

```bash
pip install roboflow
```

Train or use the pretrained model:  

```bash
python train.py --data tennis.yaml --weights yolov5s.pt --epochs 50
```

Run the analysis on a tennis video:  

```bash
python analyze.py --input input_video.mp4 --output output_video.mp4
```

---

## 📝 Project Workflow  

1. **Data Collection & Annotation**  
   - Dataset sourced from Roboflow and manually refined.  

2. **Model Training**  
   - Trained a YOLO-based model on the dataset.  

3. **Inference & Detection**  
   - Run detection on match footage.  

4. **Trajectory Tracking**  
   - Apply tracking algorithms (e.g., SORT/DeepSORT) to smooth movement.  

5. **Game Analysis**  
   - Extract ball speed, bounce points, and trajectory curves.  

---

## 🔮 Future Improvements  

- Include **racket-ball interaction analysis**.  
- Build a **dashboard with live match stats**.  
- Extend dataset with more diverse match conditions.  

---

## 🙌 Acknowledgements  

- Dataset by [Viren Dhanwani on Roboflow](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection).  
- YOLO object detection framework.  
- OpenCV for image & video processing.  

---

## 📜 License  

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.  

---

### 💡 If you like this project, please ⭐ the repo and share it with others!
