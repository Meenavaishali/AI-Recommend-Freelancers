from flask import Flask, render_template, request, jsonify
from bson import json_util
from config import db
from recommender import get_recommendations

app = Flask(__name__)

# Collection reference
freelancers_collection = db["freelancers"]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/freelancers", methods=["GET"])
def get_freelancers():
    """
    Fetch freelancers from MongoDB with optional filters.
    """
    try:
        # Get query params for filters
        skills = request.args.get("skills", "")
        min_rate = request.args.get("minRate", None, type=float)
        max_rate = request.args.get("maxRate", None, type=float)

        # Build query
        query = {}
        if skills:
            query["skills"] = {"$regex": skills, "$options": "i"}
        if min_rate is not None:
            query["hourlyRate"] = {"$gte": min_rate}
        if max_rate is not None:
            query["hourlyRate"] = {**query.get("hourlyRate", {}), "$lte": max_rate}

        # Fetch and sort freelancers
        freelancers = list(
            freelancers_collection.find(query).sort([
                ("rating", -1),
                ("sucessratio", -1),
                ("hourlyRate", 1),
                ("totalHours", 1),
            ])
        )

        return json_util.dumps(freelancers), 200

    except Exception as e:
        # Log the exception (for debugging purposes)
        print(f"Error fetching freelancers: {e}")
        return jsonify({"error": "An error occurred while fetching freelancers."}), 500


# API endpoint for recommendations
@app.route('/api/recommend', methods=['GET'])
def recommend():
    """
    Get freelancer recommendations based on user input.
    """
    try:
        # Get query parameters
        user_skills = request.args.get('skills', '')
        keyword = request.args.get('keyword', '')
        rating_str = request.args.get('rating', '0')

        # Validate the rating parameter
        try:
            rating = float(rating_str)
        except ValueError:
            return jsonify({"error": "Invalid rating value. It should be a number."}), 400

        # Call recommendation logic with the parameters
        recommendations = get_recommendations(user_skills, keyword, rating)

        return jsonify(recommendations), 200

    except ValueError as e:
        # Catching the case where float conversion fails
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    except Exception as e:
        # Log the exception (for debugging purposes)
        print(f"Error generating recommendations: {e}")
        return jsonify({"error": "An error occurred while generating recommendations."}), 500


if __name__ == "__main__":
    app.run(debug=True)
