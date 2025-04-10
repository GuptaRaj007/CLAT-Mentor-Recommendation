# Personalized Mentor Recommendation System (CLAT)

## 📌 Problem Statement

Design a simple machine learning system that recommends mentors (CLAT toppers) to CLAT aspirants based on their learning preferences, target colleges, and preparation level.

---

## 🎯 Objective

Build a recommendation engine that:
- Loads mock (or real) user and mentor data
- Extracts and processes relevant features
- Computes similarity between aspirants and mentors
- Recommends the top 3 suitable mentors

---

## 🧠 Approach Summary

- **Mock Data Generation**: Synthetic profiles for 100 aspirants and 20 mentors are created with random values for:
  - Subjects (e.g., Legal, Logical, English)
  - Colleges (NLUs)
  - Learning styles (Visual, Auditory, etc.)
  - Scores, Ratings, and Experience

- **Feature Engineering**:
  - Multi-label features like preferred subjects and target colleges are one-hot encoded.
  - Other categorical values like learning style and preparation level are also encoded.
  - Scores and ratings are normalized using `MinMaxScaler`.

- **Similarity Computation**:
  - Cosine similarity is used to match aspirants with mentors.
  - Only relevant features are considered to compute closeness in preference vectors.

- **Recommendation**:
  - For each aspirant, we retrieve the top 3 most similar mentors based on similarity scores.

---

## 🧪 File Structure

- `personalized_mentor_recommendation.py`: Python script with:
  - Mock data generation
  - Data preprocessing
  - Mentor recommendation logic
  - Example output for aspirant `A001`

- README.md: This file explaining the problem, solution, and setup.

---
✅ Important Considerations
🔹 Mock Data vs. Real Data
Current State: This project currently uses mock data generated using Python's random module to simulate aspirants and mentors.

Why It Matters: While mock data is great for prototyping, real-world applications require real anonymized data.

What Needs to Change for Real Data:

Replace the generate_mock_aspirants() and generate_mock_mentors() functions with logic to load real data from .csv, .json, or a database.

Ensure column names (like preferred_subjects, target_colleges, etc.) match real data schema.

Preprocessing must be adapted based on actual data formats (e.g., string parsing, missing values).

🔹 Basic vs. Advanced Approach
Current Method: We use cosine similarity on processed feature vectors. This is a content-based filtering technique — easy to implement and interpret.

Why It’s Used: Suitable for early-stage prototypes and smaller datasets where explainability is important.

What Could Be Improved:

Use collaborative filtering to learn from user behavior (e.g., aspirants who selected the same mentors).

Implement matrix factorization techniques like SVD or deep learning models for better personalization at scale.

Consider hybrid recommenders that combine profile features and user interaction data.

🔹 Lack of Feedback Loop (Bonus Enhancement)
Current Limitation: The script doesn't track which mentors were chosen or how useful aspirants found the recommendations.

How to Add Feedback:

Add a feedback form or button (e.g., "Was this recommendation helpful? Yes/No").

Store this interaction data in a database or CSV file.

Retrain the recommendation engine periodically based on:

Mentor click frequency

User ratings

Conversion to mentorship

## ⚙️ Setup Instructions

1. Clone the repository:
   bash
   git clone https://github.com/GuptaRaj007/personalized-mentor-recommendation.git
   cd personalized-mentor-recommendation
