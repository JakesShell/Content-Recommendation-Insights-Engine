from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, Response


app = Flask(__name__)

FEATURE_COLUMNS = [
    "Education",
    "Entertainment",
    "Technical",
    "Leadership",
    "Lifestyle",
    "Product",
    "Security",
    "Cloud",
]

content_catalog = pd.DataFrame(
    [
        {
            "content_id": 1,
            "title": "Cloud Incident Response Playbook",
            "content_type": "Guide",
            "audience": "Cloud Operations",
            "channel": "Knowledge Hub",
            "Education": 1,
            "Entertainment": 0,
            "Technical": 1,
            "Leadership": 0,
            "Lifestyle": 0,
            "Product": 0,
            "Security": 1,
            "Cloud": 1,
            "base_engagement": 88,
            "freshness": 94,
            "production_cost": 32,
            "business_goal": "Reduce incident handling time",
            "summary": "A practical guide for triaging cloud incidents, preserving evidence, and coordinating response teams.",
        },
        {
            "content_id": 2,
            "title": "AI-Powered Onboarding Checklist",
            "content_type": "Template",
            "audience": "HR And Enablement",
            "channel": "Customer Portal",
            "Education": 1,
            "Entertainment": 0,
            "Technical": 0,
            "Leadership": 1,
            "Lifestyle": 0,
            "Product": 1,
            "Security": 0,
            "Cloud": 0,
            "base_engagement": 76,
            "freshness": 87,
            "production_cost": 18,
            "business_goal": "Improve activation",
            "summary": "A ready-to-use onboarding checklist that helps new users reach value faster.",
        },
        {
            "content_id": 3,
            "title": "Behind The Dashboard: How Recommendations Work",
            "content_type": "Video",
            "audience": "Product Leaders",
            "channel": "Streaming Library",
            "Education": 1,
            "Entertainment": 1,
            "Technical": 1,
            "Leadership": 1,
            "Lifestyle": 0,
            "Product": 1,
            "Security": 0,
            "Cloud": 1,
            "base_engagement": 91,
            "freshness": 90,
            "production_cost": 61,
            "business_goal": "Increase product trust",
            "summary": "An explainer video that turns recommendation logic into a clear product story.",
        },
        {
            "content_id": 4,
            "title": "Executive Brief: Reducing Churn With Better Content Paths",
            "content_type": "Brief",
            "audience": "Executives",
            "channel": "Email Campaign",
            "Education": 1,
            "Entertainment": 0,
            "Technical": 0,
            "Leadership": 1,
            "Lifestyle": 0,
            "Product": 1,
            "Security": 0,
            "Cloud": 0,
            "base_engagement": 83,
            "freshness": 78,
            "production_cost": 22,
            "business_goal": "Reduce churn risk",
            "summary": "A board-friendly summary showing how personalized content journeys protect retention.",
        },
        {
            "content_id": 5,
            "title": "Family Learning Path: Safe Tech At Home",
            "content_type": "Course",
            "audience": "Families",
            "channel": "Learning Library",
            "Education": 1,
            "Entertainment": 1,
            "Technical": 0,
            "Leadership": 0,
            "Lifestyle": 1,
            "Product": 0,
            "Security": 1,
            "Cloud": 0,
            "base_engagement": 79,
            "freshness": 82,
            "production_cost": 35,
            "business_goal": "Increase family engagement",
            "summary": "A friendly learning path for families that introduces safe digital habits.",
        },
        {
            "content_id": 6,
            "title": "Product Launch Signal Room",
            "content_type": "Interactive",
            "audience": "Marketing Teams",
            "channel": "Launch Center",
            "Education": 0,
            "Entertainment": 1,
            "Technical": 0,
            "Leadership": 1,
            "Lifestyle": 0,
            "Product": 1,
            "Security": 0,
            "Cloud": 0,
            "base_engagement": 86,
            "freshness": 93,
            "production_cost": 46,
            "business_goal": "Improve launch engagement",
            "summary": "A high-energy interactive page that helps teams choose launch assets and audience messages.",
        },
        {
            "content_id": 7,
            "title": "Cloud Cost Optimization Mini-Series",
            "content_type": "Series",
            "audience": "Cloud Operations",
            "channel": "Video Library",
            "Education": 1,
            "Entertainment": 0,
            "Technical": 1,
            "Leadership": 0,
            "Lifestyle": 0,
            "Product": 0,
            "Security": 0,
            "Cloud": 1,
            "base_engagement": 81,
            "freshness": 84,
            "production_cost": 40,
            "business_goal": "Improve cloud cost awareness",
            "summary": "A short content series that helps teams understand cost signals and waste patterns.",
        },
        {
            "content_id": 8,
            "title": "Security Awareness Story Pack",
            "content_type": "Story Pack",
            "audience": "Employees",
            "channel": "Internal Portal",
            "Education": 1,
            "Entertainment": 1,
            "Technical": 0,
            "Leadership": 0,
            "Lifestyle": 1,
            "Product": 0,
            "Security": 1,
            "Cloud": 0,
            "base_engagement": 74,
            "freshness": 80,
            "production_cost": 24,
            "business_goal": "Improve security behavior",
            "summary": "Short scenario-based stories that make security awareness easier to remember.",
        },
        {
            "content_id": 9,
            "title": "AI Feature Adoption Map",
            "content_type": "Dashboard",
            "audience": "Product Leaders",
            "channel": "Executive Console",
            "Education": 1,
            "Entertainment": 0,
            "Technical": 1,
            "Leadership": 1,
            "Lifestyle": 0,
            "Product": 1,
            "Security": 0,
            "Cloud": 1,
            "base_engagement": 92,
            "freshness": 95,
            "production_cost": 50,
            "business_goal": "Prioritize AI adoption",
            "summary": "A dashboard concept that helps product leaders decide which AI features deserve the next sprint.",
        },
        {
            "content_id": 10,
            "title": "Customer Success Renewal Toolkit",
            "content_type": "Toolkit",
            "audience": "Customer Success",
            "channel": "Customer Portal",
            "Education": 1,
            "Entertainment": 0,
            "Technical": 0,
            "Leadership": 1,
            "Lifestyle": 0,
            "Product": 1,
            "Security": 0,
            "Cloud": 0,
            "base_engagement": 85,
            "freshness": 86,
            "production_cost": 28,
            "business_goal": "Protect renewals",
            "summary": "A renewal-focused toolkit that helps customer teams recommend the right next resource.",
        },
        {
            "content_id": 11,
            "title": "Modern Cloud Security Architecture Walkthrough",
            "content_type": "Webinar",
            "audience": "Security Leaders",
            "channel": "Live Event",
            "Education": 1,
            "Entertainment": 0,
            "Technical": 1,
            "Leadership": 1,
            "Lifestyle": 0,
            "Product": 0,
            "Security": 1,
            "Cloud": 1,
            "base_engagement": 89,
            "freshness": 91,
            "production_cost": 58,
            "business_goal": "Build security authority",
            "summary": "A technical webinar showing how cloud teams design secure operating environments.",
        },
        {
            "content_id": 12,
            "title": "Creator Growth Lab: Content Experiments That Convert",
            "content_type": "Playbook",
            "audience": "Marketing Teams",
            "channel": "Growth Hub",
            "Education": 1,
            "Entertainment": 1,
            "Technical": 0,
            "Leadership": 0,
            "Lifestyle": 1,
            "Product": 1,
            "Security": 0,
            "Cloud": 0,
            "base_engagement": 87,
            "freshness": 92,
            "production_cost": 30,
            "business_goal": "Improve conversion",
            "summary": "A practical playbook for testing content ideas and turning engagement into pipeline.",
        },
    ]
)

user_activity = pd.DataFrame(
    [
        {"user_id": 201, "name": "Ari CloudOps", "segment": "Cloud Operations", "content_id": 1, "rating": 5.0, "completion": 94},
        {"user_id": 201, "name": "Ari CloudOps", "segment": "Cloud Operations", "content_id": 7, "rating": 4.5, "completion": 82},
        {"user_id": 201, "name": "Ari CloudOps", "segment": "Cloud Operations", "content_id": 11, "rating": 4.8, "completion": 88},
        {"user_id": 202, "name": "Maya Product", "segment": "Product Leaders", "content_id": 3, "rating": 4.7, "completion": 86},
        {"user_id": 202, "name": "Maya Product", "segment": "Product Leaders", "content_id": 9, "rating": 5.0, "completion": 95},
        {"user_id": 202, "name": "Maya Product", "segment": "Product Leaders", "content_id": 4, "rating": 4.2, "completion": 74},
        {"user_id": 203, "name": "Leo Growth", "segment": "Marketing Teams", "content_id": 6, "rating": 4.8, "completion": 89},
        {"user_id": 203, "name": "Leo Growth", "segment": "Marketing Teams", "content_id": 12, "rating": 4.9, "completion": 93},
        {"user_id": 203, "name": "Leo Growth", "segment": "Marketing Teams", "content_id": 2, "rating": 3.7, "completion": 62},
        {"user_id": 204, "name": "Nia Security", "segment": "Security Leaders", "content_id": 11, "rating": 5.0, "completion": 96},
        {"user_id": 204, "name": "Nia Security", "segment": "Security Leaders", "content_id": 1, "rating": 4.6, "completion": 85},
        {"user_id": 204, "name": "Nia Security", "segment": "Security Leaders", "content_id": 8, "rating": 4.0, "completion": 71},
        {"user_id": 205, "name": "Sage Learner", "segment": "Families", "content_id": 5, "rating": 4.9, "completion": 97},
        {"user_id": 205, "name": "Sage Learner", "segment": "Families", "content_id": 8, "rating": 4.5, "completion": 84},
        {"user_id": 205, "name": "Sage Learner", "segment": "Families", "content_id": 2, "rating": 4.1, "completion": 73},
    ]
)


@dataclass
class RecommendationResult:
    audience: Dict
    watched: List[Dict]
    recommendations: List[Dict]
    profile: Dict
    metrics: Dict
    ai_brief: Dict
    clusters: List[Dict]


def build_user_profile(user_id: int) -> Tuple[pd.DataFrame | None, pd.Series | None]:
    history = user_activity[user_activity["user_id"] == user_id]

    if history.empty:
        return None, None

    merged = history.merge(content_catalog, on="content_id", how="left")
    weights = merged["rating"] * (merged["completion"] / 100)
    weighted_matrix = merged[FEATURE_COLUMNS].multiply(weights, axis=0)
    profile = weighted_matrix.sum() / weights.sum()

    return merged, profile


def explain_recommendation(row: pd.Series, profile: pd.Series) -> List[str]:
    matching_features = [
        feature for feature in FEATURE_COLUMNS
        if row[feature] == 1 and profile[feature] >= profile.mean()
    ]

    reasons = []

    if matching_features:
        reasons.append("Matches strong audience interests: " + ", ".join(matching_features[:3]))

    if row["freshness"] >= 90:
        reasons.append("Fresh content with strong timing signal")

    if row["base_engagement"] >= 88:
        reasons.append("High historical engagement potential")

    if row["production_cost"] <= 30:
        reasons.append("Efficient to promote with low production burden")

    if not reasons:
        reasons.append("Balanced fit across audience profile and catalog signals")

    return reasons


def classify_priority(score: float) -> str:
    if score >= 85:
        return "Priority 1"
    if score >= 72:
        return "Priority 2"
    if score >= 55:
        return "Priority 3"
    return "Backlog"


def build_recommendations(user_id: int, goal: str = "retention", top_n: int = 5) -> RecommendationResult | None:
    watched, profile = build_user_profile(user_id)

    if watched is None or profile is None:
        return None

    watched_ids = set(watched["content_id"].tolist())
    unseen = content_catalog[~content_catalog["content_id"].isin(watched_ids)].copy()

    affinity = unseen[FEATURE_COLUMNS].dot(profile.values)
    engagement = unseen["base_engagement"] / 100
    freshness = unseen["freshness"] / 100
    cost_efficiency = 1 - (unseen["production_cost"] / 100)

    goal_boost = {
        "retention": unseen["Education"] * 0.07 + unseen["Leadership"] * 0.05,
        "activation": unseen["Product"] * 0.08 + unseen["Education"] * 0.05,
        "growth": unseen["Entertainment"] * 0.06 + unseen["Product"] * 0.07,
        "trust": unseen["Security"] * 0.08 + unseen["Cloud"] * 0.06,
    }.get(goal, unseen["Education"] * 0.04)

    raw_score = (
        affinity * 0.42 +
        engagement * 0.25 +
        freshness * 0.18 +
        cost_efficiency * 0.10 +
        goal_boost
    )

    # Convert the raw affinity score into a realistic executive-facing percentage.
    # This keeps recommendations ranked while avoiding fake-looking 100% scores.
    min_score = float(raw_score.min())
    max_score = float(raw_score.max())

    if max_score == min_score:
        unseen["recommendation_score"] = 78.0
    else:
        normalized = (raw_score - min_score) / (max_score - min_score)
        unseen["recommendation_score"] = 58 + (normalized * 36)

    unseen["recommendation_score"] = unseen["recommendation_score"].clip(42, 94)
    unseen["priority"] = unseen["recommendation_score"].apply(classify_priority)
    unseen["forecast_lift"] = np.round(unseen["recommendation_score"] * 0.34 + unseen["base_engagement"] * 0.21).astype(int)
    unseen["retention_signal"] = np.where(unseen["recommendation_score"] >= 78, "Strong", np.where(unseen["recommendation_score"] >= 60, "Moderate", "Low"))
    unseen["cloud_ready_status"] = np.where(unseen["content_type"].isin(["Dashboard", "Guide", "Webinar", "Series"]), "API Ready", "Queue Ready")
    unseen["why"] = unseen.apply(lambda row: explain_recommendation(row, profile), axis=1)

    ranked = unseen.sort_values(by="recommendation_score", ascending=False).head(top_n)

    profile_summary = (
        pd.DataFrame({"signal": FEATURE_COLUMNS, "score": np.round(profile.values * 100, 1)})
        .sort_values(by="score", ascending=False)
        .to_dict(orient="records")
    )

    watched_records = watched[["title", "content_type", "rating", "completion", "business_goal"]].to_dict(orient="records")
    recommendations = ranked[
        [
            "content_id",
            "title",
            "content_type",
            "audience",
            "channel",
            "recommendation_score",
            "priority",
            "forecast_lift",
            "retention_signal",
            "cloud_ready_status",
            "business_goal",
            "summary",
            "why",
        ]
    ].to_dict(orient="records")

    audience_row = user_activity[user_activity["user_id"] == user_id].iloc[0]
    audience = {
        "user_id": int(user_id),
        "name": audience_row["name"],
        "segment": audience_row["segment"],
        "goal": goal.title(),
    }

    total_forecast = int(sum(item["forecast_lift"] for item in recommendations))
    top_score = float(recommendations[0]["recommendation_score"]) if recommendations else 0
    risk = "Low" if top_score >= 75 else "Moderate" if top_score >= 55 else "High"

    metrics = {
        "recommended_items": len(recommendations),
        "forecast_lift": total_forecast,
        "audience_fit": round(top_score, 1),
        "retention_risk": risk,
        "avg_completion": round(float(watched["completion"].mean()), 1),
        "profile_strength": round(float(profile.max() * 100), 1),
    }

    strongest_signal = profile_summary[0]["signal"]
    best_item = recommendations[0] if recommendations else None

    ai_brief = {
        "headline": f"Launch {best_item['title'] if best_item else 'next-best content'} first",
        "summary": (
            f"{audience['name']} shows strongest affinity for {strongest_signal}. "
            f"The highest-ranked recommendation supports {best_item['business_goal'] if best_item else 'audience engagement'} "
            f"and is forecast to create {best_item['forecast_lift'] if best_item else 0} engagement lift points."
        ),
        "next_steps": [
            "Promote the top recommendation in the next content slot.",
            "Pair it with one supporting resource from the same audience theme.",
            "Monitor completion and click-through before expanding the content path.",
            "Route high-performing content into the cloud publishing pipeline.",
        ],
    }

    clusters = []
    for feature in FEATURE_COLUMNS:
        matching = content_catalog[content_catalog[feature] == 1]
        clusters.append(
            {
                "name": feature,
                "items": int(len(matching)),
                "avg_engagement": round(float(matching["base_engagement"].mean()), 1),
                "signal": round(float(profile[feature] * 100), 1),
            }
        )

    clusters = sorted(clusters, key=lambda item: item["signal"], reverse=True)

    return RecommendationResult(
        audience=audience,
        watched=watched_records,
        recommendations=recommendations,
        profile=profile_summary,
        metrics=metrics,
        ai_brief=ai_brief,
        clusters=clusters,
    )


def result_to_dict(result: RecommendationResult) -> Dict:
    return {
        "audience": result.audience,
        "watched": result.watched,
        "recommendations": result.recommendations,
        "profile": result.profile,
        "metrics": result.metrics,
        "ai_brief": result.ai_brief,
        "clusters": result.clusters,
    }


@app.route("/")
def index():
    audiences = (
        user_activity[["user_id", "name", "segment"]]
        .drop_duplicates()
        .sort_values(by="user_id")
        .to_dict(orient="records")
    )

    default_result = build_recommendations(202, "retention", 6)

    return render_template(
        "index.html",
        audiences=audiences,
        initial=result_to_dict(default_result),
        current_year=datetime.now().year,
    )


@app.route("/api/recommendations")
def api_recommendations():
    user_id = int(request.args.get("user_id", 202))
    goal = request.args.get("goal", "retention")
    result = build_recommendations(user_id, goal, 6)

    if result is None:
        return jsonify({"error": "Audience not found"}), 404

    return jsonify(result_to_dict(result))


@app.route("/api/export")
def export_csv():
    user_id = int(request.args.get("user_id", 202))
    goal = request.args.get("goal", "retention")
    result = build_recommendations(user_id, goal, 12)

    if result is None:
        return Response("Audience not found", status=404)

    rows = []
    for item in result.recommendations:
        rows.append(
            {
                "audience": result.audience["name"],
                "segment": result.audience["segment"],
                "goal": result.audience["goal"],
                "title": item["title"],
                "content_type": item["content_type"],
                "score": round(item["recommendation_score"], 1),
                "priority": item["priority"],
                "forecast_lift": item["forecast_lift"],
                "retention_signal": item["retention_signal"],
                "business_goal": item["business_goal"],
            }
        )

    df = pd.DataFrame(rows)
    csv_data = df.to_csv(index=False)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=pulsecue_recommendation_plan.csv"
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
