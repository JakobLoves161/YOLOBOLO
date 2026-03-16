# 👕 YOLO Clothing Detection Web App

This project is a **Streamlit web application** that uses **YOLOv8** to detect and classify clothing items in images.

Users can upload an image and the AI model will detect clothing items such as:

* T-Shirts
* Jackets
* Pants
* Dresses
* Skirts
* Hoodies
* Shoes

The application draws **bounding boxes** around detected clothing items and displays the prediction results.

---

# 🚀 Features

* Upload images directly in the browser
* Real-time clothing detection using YOLOv8
* Bounding boxes around detected objects
* Built with Streamlit for a simple web interface
* Ready to deploy on Streamlit Cloud

---

# 🧠 Model

This project uses the **YOLOv8 object detection model** provided by Ultralytics.

The model detects objects in images and returns:

* detected class
* confidence score
* bounding box

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/yolobolo.git
cd yolobolo
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```
yolobolo
│
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# 🌐 Deploy on Streamlit Cloud

1. Push the repository to GitHub
2. Go to Streamlit Cloud
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy the app

---

# 🛠 Requirements

* Python 3.11
* Streamlit
* Ultralytics YOLO
* OpenCV (headless version)
* PyTorch

---

# 📚 Technologies Used

* Streamlit
* YOLOv8
* PyTorch
* OpenCV
* NumPy
* Pillow

---

#
