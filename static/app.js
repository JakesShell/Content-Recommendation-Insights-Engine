const state = {
    userId: window.__INITIAL_DATA__.audience.user_id,
    goal: "retention",
    data: window.__INITIAL_DATA__,
};

const elements = {
    form: document.getElementById("recommendationForm"),
    audienceSelect: document.getElementById("audienceSelect"),
    goalSelect: document.getElementById("goalSelect"),
    modeToggle: document.getElementById("modeToggle"),
    heroFit: document.getElementById("heroFit"),
    heroRecommendation: document.getElementById("heroRecommendation"),
    heroSegment: document.getElementById("heroSegment"),
    metricFit: document.getElementById("metricFit"),
    metricLift: document.getElementById("metricLift"),
    metricItems: document.getElementById("metricItems"),
    metricRisk: document.getElementById("metricRisk"),
    profileStrength: document.getElementById("profileStrength"),
    briefHeadline: document.getElementById("briefHeadline"),
    briefSummary: document.getElementById("briefSummary"),
    briefSteps: document.getElementById("briefSteps"),
    tasteBars: document.getElementById("tasteBars"),
    recommendationGrid: document.getElementById("recommendationGrid"),
    historyList: document.getElementById("historyList"),
    clusterList: document.getElementById("clusterList"),
    exportLink: document.getElementById("exportLink"),
};

function formatScore(value) {
    return `${Number(value).toFixed(1)}%`;
}

function setLoading(isLoading) {
    const button = elements.form.querySelector("button");
    button.disabled = isLoading;
    button.textContent = isLoading ? "Reading Signals..." : "Generate Recommendation Plan";
}

async function fetchRecommendations() {
    setLoading(true);

    const params = new URLSearchParams({
        user_id: state.userId,
        goal: state.goal,
    });

    const response = await fetch(`/api/recommendations?${params.toString()}`);

    if (!response.ok) {
        setLoading(false);
        alert("Could not generate recommendations for this audience.");
        return;
    }

    state.data = await response.json();
    renderDashboard();
    setLoading(false);
}

function renderDashboard() {
    const data = state.data;

    elements.heroFit.textContent = formatScore(data.metrics.audience_fit);
    elements.heroRecommendation.textContent = data.ai_brief.headline;
    elements.heroSegment.textContent = `${data.audience.name} · ${data.audience.segment}`;

    elements.metricFit.textContent = formatScore(data.metrics.audience_fit);
    elements.metricLift.textContent = data.metrics.forecast_lift;
    elements.metricItems.textContent = data.metrics.recommended_items;
    elements.metricRisk.textContent = data.metrics.retention_risk;
    elements.profileStrength.textContent = `${data.metrics.profile_strength}% strongest signal`;

    elements.briefHeadline.textContent = data.ai_brief.headline;
    elements.briefSummary.textContent = data.ai_brief.summary;

    elements.briefSteps.innerHTML = data.ai_brief.next_steps
        .map((step) => `<li>${step}</li>`)
        .join("");

    elements.tasteBars.innerHTML = data.profile
        .map((item) => `
            <div class="taste-row">
                <span>${item.signal}</span>
                <i style="--w: ${item.score}%"></i>
                <strong>${item.score}%</strong>
            </div>
        `)
        .join("");

    elements.recommendationGrid.innerHTML = data.recommendations
        .map((item) => `
            <article class="recommendation-card">
                <div class="card-topline">
                    <span>${item.priority}</span>
                    <b>${Number(item.recommendation_score).toFixed(1)}%</b>
                </div>
                <h3>${item.title}</h3>
                <p>${item.summary}</p>
                <div class="card-tags">
                    <span>${item.content_type}</span>
                    <span>${item.channel}</span>
                    <span>${item.cloud_ready_status}</span>
                </div>
                <div class="why-box">
                    <strong>Why PulseCue recommends it</strong>
                    <ul>
                        ${item.why.map((reason) => `<li>${reason}</li>`).join("")}
                    </ul>
                </div>
            </article>
        `)
        .join("");

    elements.historyList.innerHTML = data.watched
        .map((item) => `
            <div>
                <strong>${item.title}</strong>
                <span>${item.content_type} · Rating ${item.rating} · ${item.completion}% complete</span>
            </div>
        `)
        .join("");

    elements.clusterList.innerHTML = data.clusters
        .slice(0, 5)
        .map((cluster) => `
            <div>
                <span>${cluster.name}</span>
                <strong>${cluster.signal}%</strong>
                <small>${cluster.items} items · ${cluster.avg_engagement} avg engagement</small>
            </div>
        `)
        .join("");

    elements.exportLink.href = `/api/export?user_id=${state.userId}&goal=${state.goal}`;

    document.querySelectorAll(".recommendation-card, .metric-grid article, .brief-card, .taste-card")
        .forEach((card, index) => {
            card.animate(
                [
                    { transform: "translateY(10px)", opacity: 0.66 },
                    { transform: "translateY(0)", opacity: 1 },
                ],
                {
                    duration: 360 + index * 30,
                    easing: "ease-out",
                }
            );
        });
}

elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    state.userId = elements.audienceSelect.value;
    state.goal = elements.goalSelect.value;
    fetchRecommendations();
});

elements.audienceSelect.addEventListener("change", () => {
    state.userId = elements.audienceSelect.value;
    fetchRecommendations();
});

elements.goalSelect.addEventListener("change", () => {
    state.goal = elements.goalSelect.value;
    fetchRecommendations();
});

elements.modeToggle.addEventListener("click", () => {
    document.body.classList.toggle("light");
    elements.modeToggle.textContent = document.body.classList.contains("light") ? "Dark Mode" : "Light Mode";
});

renderDashboard();


/* =========================================================
   CTA ACTIONS
   Makes the new CTA areas feel active and useful.
   ========================================================= */

document.addEventListener("click", (event) => {
    const generateButton = event.target.closest('[data-action="generate-plan"]');
    if (generateButton) {
        event.preventDefault();
        elements.form.requestSubmit();
        document.getElementById("recommendations")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    const switchAudienceButton = event.target.closest('[data-action="switch-audience"]');
    if (switchAudienceButton) {
        event.preventDefault();

        const options = Array.from(elements.audienceSelect.options);
        const currentIndex = options.findIndex(option => option.value === elements.audienceSelect.value);
        const nextIndex = (currentIndex + 1) % options.length;

        elements.audienceSelect.value = options[nextIndex].value;
        state.userId = elements.audienceSelect.value;
        fetchRecommendations();

        document.getElementById("signal-room")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
});

function syncCtaExportLink() {
    const ctaExportLink = document.getElementById("ctaExportLink");
    if (ctaExportLink) {
        ctaExportLink.href = `/api/export?user_id=${state.userId}&goal=${state.goal}`;
    }
}

const originalRenderDashboard = renderDashboard;
renderDashboard = function patchedRenderDashboard() {
    originalRenderDashboard();
    syncCtaExportLink();
};

syncCtaExportLink();
