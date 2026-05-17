import pandas as pd
import numpy as np

content_catalog = pd.DataFrame([
    {"content_id": 1, "title": "City Heat", "Action": 1, "Drama": 0, "Comedy": 0, "Documentary": 0, "SciFi": 0, "Family": 0},
    {"content_id": 2, "title": "Quiet Harbor", "Action": 0, "Drama": 1, "Comedy": 0, "Documentary": 0, "SciFi": 0, "Family": 0},
    {"content_id": 3, "title": "Office Sparks", "Action": 0, "Drama": 0, "Comedy": 1, "Documentary": 0, "SciFi": 0, "Family": 0},
    {"content_id": 4, "title": "Planet Signal", "Action": 0, "Drama": 0, "Comedy": 0, "Documentary": 0, "SciFi": 1, "Family": 0},
    {"content_id": 5, "title": "Nature Code", "Action": 0, "Drama": 0, "Comedy": 0, "Documentary": 1, "SciFi": 0, "Family": 0},
    {"content_id": 6, "title": "Weekend Heroes", "Action": 1, "Drama": 0, "Comedy": 1, "Documentary": 0, "SciFi": 0, "Family": 0},
    {"content_id": 7, "title": "Signal Home", "Action": 0, "Drama": 1, "Comedy": 0, "Documentary": 0, "SciFi": 1, "Family": 0},
    {"content_id": 8, "title": "Family Trail", "Action": 0, "Drama": 0, "Comedy": 0, "Documentary": 0, "SciFi": 0, "Family": 1},
    {"content_id": 9, "title": "Market Minds", "Action": 0, "Drama": 0, "Comedy": 0, "Documentary": 1, "SciFi": 0, "Family": 0},
    {"content_id": 10, "title": "Laugh Circuit", "Action": 0, "Drama": 0, "Comedy": 1, "Documentary": 0, "SciFi": 1, "Family": 0}
])

user_ratings = pd.DataFrame([
    {"user_id": 101, "content_id": 1, "rating": 4.5},
    {"user_id": 101, "content_id": 4, "rating": 5.0},
    {"user_id": 101, "content_id": 7, "rating": 4.0},
    {"user_id": 102, "content_id": 2, "rating": 4.5},
    {"user_id": 102, "content_id": 5, "rating": 4.0},
    {"user_id": 102, "content_id": 9, "rating": 4.5},
    {"user_id": 103, "content_id": 3, "rating": 4.0},
    {"user_id": 103, "content_id": 6, "rating": 4.5},
    {"user_id": 103, "content_id": 10, "rating": 4.0},
    {"user_id": 104, "content_id": 8, "rating": 5.0},
    {"user_id": 104, "content_id": 2, "rating": 3.5},
    {"user_id": 104, "content_id": 5, "rating": 4.0}
])

feature_columns = ["Action", "Drama", "Comedy", "Documentary", "SciFi", "Family"]


def build_user_profile(user_id):
    history = user_ratings[user_ratings["user_id"] == user_id]

    if history.empty:
        return None, None

    merged = history.merge(content_catalog, on="content_id", how="left")
    weighted_matrix = merged[feature_columns].multiply(merged["rating"], axis=0)
    profile = weighted_matrix.sum() / merged["rating"].sum()

    return merged, profile


def get_recommendations(user_id, top_n=5):
    watched, profile = build_user_profile(user_id)

    if watched is None:
        return None, None, None

    watched_ids = set(watched["content_id"].tolist())
    unseen = content_catalog[~content_catalog["content_id"].isin(watched_ids)].copy()

    unseen["recommendation_score"] = unseen[feature_columns].dot(profile.values)
    unseen = unseen.sort_values(by="recommendation_score", ascending=False).head(top_n)

    return watched, profile, unseen[["title", "recommendation_score"]]


def print_available_users():
    ids = sorted(user_ratings["user_id"].unique().tolist())
    print("Available Sample Users:", ", ".join(str(x) for x in ids))


def main():
    print("Content Recommendation Insights Engine")
    print("Sample recommendation workflow for catalog and content teams")
    print()

    print_available_users()
    print()

    try:
        user_id = int(input("Enter a user ID: ").strip())
    except ValueError:
        print("Invalid user ID. Please enter a number.")
        return

    watched, profile, recommendations = get_recommendations(user_id)

    if watched is None:
        print("User not found in the sample dataset.")
        return

    print()
    print(f"Viewing recommendation profile for User {user_id}")
    print()

    print("Previously Rated Content")
    print(watched[["title", "rating"]].to_string(index=False))
    print()

    profile_summary = pd.DataFrame({
        "Preference Area": feature_columns,
        "Score": np.round(profile.values, 3)
    }).sort_values(by="Score", ascending=False)

    print("Preference Summary")
    print(profile_summary.to_string(index=False))
    print()

    print("Top Recommendations")
    recommendations["recommendation_score"] = recommendations["recommendation_score"].round(3)
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
