# Student Performance Prediction

This project predicts a student's expected score and grade from three inputs:

- Weekly self-study hours
- Attendance percentage
- Class participation

It has a FastAPI backend for ML predictions and a React + Vite frontend for the user interface.

## Features

- Predicts student score and grade from study, attendance, and participation inputs
- Shows performance band, risk level, and focus area
- Includes preset student profiles for quick testing
- Displays engagement, consistency, and support-need insights
- Gives strengths and improvement recommendations after each prediction
- Uses a trained machine learning backend instead of hardcoded frontend results

## Project Structure

```text
Student Performance/
+-- backend/
|   +-- main.py          # FastAPI app
|   +-- train.py         # Model training script
|   +-- utils.py         # Feature engineering helpers
|   +-- model/           # Saved trained models
+-- data/
|   +-- student_performance (1).csv
+-- frontend/
|   +-- Student Performance/
|       +-- src/
+-- requirements.txt
+-- README.md
```

## Backend Setup

From the project root:

```powershell
pip install -r requirements.txt
cd backend
uvicorn main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Useful API routes:

```text
GET  /
GET  /presets
POST /predict
```

## Frontend Setup

Open a second terminal:

```powershell
npm install
npm run dev
```

The frontend will usually run at:

```text
http://localhost:5173
```

Keep the backend running while using the frontend.

## Frontend UI Details

The frontend is built with React and Vite. It gives users a clean dashboard-style interface for testing student performance predictions.

Main UI sections:

- Header area with the project title and short description
- Preset cards for quick student profiles:
  - High Achiever
  - Balanced Learner
  - Needs Support
- Input controls for:
  - Weekly self-study hours
  - Attendance percentage
  - Class participation
- Prediction button that sends the form data to the FastAPI backend
- Score outlook panel showing:
  - Predicted score
  - Predicted grade
  - Performance band
  - Risk level
  - Focus area
- Insight metrics showing:
  - Engagement score
  - Consistency index
  - Support need index
- Strengths and recommendations generated from the prediction result
- Error handling message if the backend is not running

The frontend calls the backend API at:

```text
http://127.0.0.1:8000
```

This API base URL is defined in:

```text
frontend/Student Performance/src/App.jsx
```

## Retrain The Models

If you update the dataset or want to regenerate the model files:

```powershell
python backend/train.py
```

The training script saves:

```text
backend/model/score_model.pkl
backend/model/grade_model.pkl
backend/model/confusion_matrix.png
```

The training script uses a representative sample from the large CSV so retraining stays practical and model files stay small.

## Model Notes

Exact grade accuracy is limited because the model only receives three input features, but it predicts several grade classes:

```text
A+, A, B+, B, C+, C, D, F
```

The predicted score and performance band are usually more useful than treating the exact grade as perfect. Accuracy can improve if more academic features are added, such as previous scores, assignment results, exam history, or other learning indicators.
