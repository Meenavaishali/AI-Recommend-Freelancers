import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
from config import db
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load a lightweight embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Error loading embedding model: {e}")
    embedding_model = None 


def get_freelancer_data():
    """
    Fetch freelancer data from MongoDB.
    """
    try:
        freelancers_collection = db["freelancers"]
        freelancers = list(freelancers_collection.find())
        return freelancers
    except Exception as e:
        logger.error(f"Error fetching freelancer data: {e}")
        return []


def preprocess_freelancer_data(freelancers):
    """
    Combine skills and title for embedding and similarity calculations.
    """
    try:
        return [
            f"{freelancer['title']} {freelancer['skills']}" for freelancer in freelancers
        ]
    except KeyError as e:
        logger.error(f"Missing key in freelancer data: {e}")
        return [] 


def precompute_embeddings(freelancer_texts):
    """
    Precompute and normalize embeddings for freelancer texts.
    """
    if embedding_model is None:
        logger.error("Embedding model not loaded, skipping embedding computation.")
        return np.array([])  # Return empty array if the model isn't loaded
    
    try:
        embeddings = embedding_model.encode(freelancer_texts, batch_size=32, show_progress_bar=True)
        return normalize(embeddings)
    except Exception as e:
        logger.error(f"Error in embedding computation: {e}")
        return np.array([])


def build_ann_index(embeddings):
    """
    Build an Approximate Nearest Neighbor (ANN) index using Faiss.
    """
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index
    except Exception as e:
        logger.error(f"Error building ANN index: {e}")
        return None


def normalize_success_ratio(success_ratio_str):
    """
    Normalize success ratio string (e.g., "97%") to float (e.g., 0.97).
    """
    try:
        return float(success_ratio_str.replace('%', '')) / 100
    except ValueError as e:
        logger.error(f"Error normalizing success ratio: {e}")
        return 0.0


def sort_freelancers(freelancers, similarity_scores):
    """
    Sort freelancers based on similarity score, rating, success ratio,
    hourly rate (lower is better), and total hours (lower is better).
    """
    try:
        for freelancer in freelancers:
            freelancer["normalized_success_ratio"] = normalize_success_ratio(
                freelancer.get("sucessratio", "0%")
            )
            freelancer["rating"] = freelancer.get("rating", 0)
            freelancer["hourlyRate"] = freelancer.get("hourlyRate", float("inf"))
            freelancer["totalHours"] = freelancer.get("totalHours", 0)

        # Sort freelancers using combined criteria
        sorted_freelancers = sorted(
            zip(freelancers, similarity_scores),
            key=lambda x: (
                -x[1],  # Higher similarity score
                -x[0]["rating"],  # Higher rating
                -x[0]["normalized_success_ratio"],  # Higher success ratio
                x[0]["hourlyRate"],  # Lower hourly rate
                x[0]["totalHours"],  # Lower total hours
            ),
        )
        return [freelancer[0] for freelancer in sorted_freelancers]
    except Exception as e:
        logger.error(f"Error sorting freelancers: {e}")
        return freelancers 


def get_recommendations(user_input_skills, user_input_title, rating):
    """
    Generate recommendations based on user input, optimized for performance.
    """
    try:
        freelancers = get_freelancer_data()

        # Combine skills and title into a single text for each freelancer
        freelancer_texts = preprocess_freelancer_data(freelancers)

        # Precompute embeddings for freelancers (only needed once, cache these in production)
        precomputed_embeddings = precompute_embeddings(freelancer_texts)
        if precomputed_embeddings.size == 0:
            logger.error("Embedding computation failed, skipping recommendation generation.")
            return []  # Return empty list if embeddings are empty
        
        ann_index = build_ann_index(precomputed_embeddings)
        if ann_index is None:
            logger.error("ANN index creation failed, skipping recommendation generation.")
            return []  # Return empty list if ANN index creation failed

        # Combine user input skills and title
        user_input = f"{user_input_title} {user_input_skills} rating:{rating}"
        user_embedding = embedding_model.encode([user_input])
        user_embedding_normalized = normalize(user_embedding)

        # Perform ANN search for top N matches
        top_k = 10  # Number of top matches to retrieve
        distances, indices = ann_index.search(user_embedding_normalized, top_k)

        # Fetch top-matched freelancers
        matched_freelancers = [freelancers[i] for i in indices.flatten()]
        similarity_scores = distances.flatten()

        # Filter freelancers based on rating
        filtered_freelancers = [
            freelancer for freelancer in matched_freelancers if freelancer.get("rating", 0) >= rating
        ]

        # If fewer than 5 freelancers match, return all of them
        if len(filtered_freelancers) < 5:
            sorted_freelancers = sort_freelancers(filtered_freelancers, similarity_scores[:len(filtered_freelancers)])
        else:
            # Sort matched freelancers based on multiple criteria
            sorted_freelancers = sort_freelancers(filtered_freelancers[:5], similarity_scores[:5])

        # Convert all types to JSON-serializable formats
        def to_serializable(freelancer, score):
            return {
                "name": freelancer["name"],
                "title": freelancer["title"],
                "description": freelancer["description"],
                "skills": freelancer["skills"],
                "hourlyRate": float(freelancer["hourlyRate"]),
                "totalHours": int(freelancer["totalHours"]),
                "totalJobs": int(freelancer["totalJobs"]),
                "rating": float(freelancer["rating"]),
                "sucessratio": freelancer["sucessratio"],
                "similarity_score": float(score),
            }

        return [to_serializable(freelancer, score) for freelancer, score in list(zip(sorted_freelancers, similarity_scores))[:5]]

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []
