# AI Freelancer Recommendations

AI-Freelancer-Recommendations is a web-based application that helps users find the most suitable freelancers based on their skills, title, rating, and other criteria. It leverages advanced machine learning techniques like **Sentence Transformers** for semantic matching and **Faiss** for fast Approximate Nearest Neighbor (ANN) search. The application is built with Flask, MongoDB, and a simple HTML frontend.

---

## Features

- **Freelancer Recommendation System**: Matches freelancers to user-provided requirements using AI-based similarity scoring.
- **Dynamic Filters**: Search freelancers by skills, hourly rate range, and other attributes.
- **Precomputed Embeddings**: Optimizes performance by precomputing freelancer embeddings.
- **Ranking Algorithm**: Ranks freelancers based on multiple weighted criteria such as rating, success ratio, hourly rate, and total hours.
- **FLASK**: Exposes endpoints for fetching freelancer data and generating recommendations.
- **Lightweight Frontend**: Intuitive interface for entering search criteria and viewing recommendations.

## Tech Stack

- **Backend**: Flask, MongoDB, Faiss, Sentence Transformers
- **Frontend**: HTML, CSS
- **Libraries Used**:
  - `SentenceTransformers` for text embeddings
  - `Faiss` for ANN index and similarity searches
  - `Flask` for API and web server
  - `pymongo` for MongoDB interaction
  - `numpy` and `scikit-learn` for data processing

## Project Structure

AI-Freelancer-Recommendations/
│
├── config/
│   ├── __init__.py          # Makes the config folder a package
│   ├── db.py                # MongoDB connection and configuration
│
├── templates/               # HTML templates for rendering UI
│   └── index.html           # Main HTML file for the application
│
├── recommender/             # Core recommendation logic
│   ├── __init__.py          # Makes the recommender folder a package
│   └── recommender.py       # Logic for recommendation generation
├── requirements.txt         # Python dependencies for the project
├── README.md                # Project documentation

## How It Works

1. **Data Retrieval**: Freelancer data is fetched from a MongoDB collection.
2. **Text Embeddings**: Skills and titles are encoded using `SentenceTransformers` into vector embeddings.
3. **Similarity Search**: The input text is compared against the precomputed embeddings using Faiss for ANN.
4. **Ranking**: Freelancers are sorted based on a weighted score combining:
   - Similarity score
   - Rating
   - Success ratio
   - Hourly rate (lower is better)
   - Total hours worked (lower is better)
5. **Recommendations**: Top freelancers are returned as a JSON response or displayed in the frontend.

---

## API Endpoints

### 1. **Homepage**
   - **URL**: `/`
   - **Method**: `GET`
   - **Description**: Renders the main application interface.

### 2. **Fetch Freelancers**
   - **URL**: `/api/freelancers`
   - **Method**: `GET`
   - **Query Parameters**:
     - `skills` (optional): Filter by skills (e.g., `Python`).
     - `minRate` (optional): Minimum hourly rate.
     - `maxRate` (optional): Maximum hourly rate.
   - **Response**: List of freelancers matching the filters.

### 3. **Generate Recommendations**
   - **URL**: `/api/recommend`
   - **Method**: `GET`
   - **Query Parameters**:
     - `skills`: Required skills (e.g., `Python, AI`).
     - `keyword`: Title or additional keywords.
     - `rating`: Minimum acceptable rating.
   - **Response**: Top 5 freelancer recommendations with similarity scores.

---

## Installation

1. Clone the repository:~

   `git clone https://github.com/your-username/AI-Freelancer-Recommendations.git`
   `cd AI-Freelancer-Recommendations`


