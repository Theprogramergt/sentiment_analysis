const API_BASE_URL = (window.MOVIE_API_URL || "/api").replace(/\/$/, "");
const MAX_REVIEW_LENGTH = 2000;

const reviewInput = document.querySelector("#review-input");
const charCount = document.querySelector("#char-count");
const analyzeButton = document.querySelector("#analyze-btn");
const errorMessage = document.querySelector("#error-message");
const connectionStatus = document.querySelector("#connection-status");
const connectionLabel = document.querySelector(".connection-label");
const resultsEmpty = document.querySelector("#results-empty");
const resultsContent = document.querySelector("#results-content");

function setConnectionStatus(online) {
  connectionStatus.classList.toggle("is-online", online);
  connectionStatus.classList.toggle("is-offline", !online);
  connectionLabel.textContent = online ? "API connected" : "API unavailable";
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

function updateCharacterCount() {
  charCount.textContent = `${reviewInput.value.length.toLocaleString()} / ${MAX_REVIEW_LENGTH.toLocaleString()}`;
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading;
  analyzeButton.classList.toggle("loading", isLoading);
  analyzeButton.querySelector(".button-label").textContent = isLoading ? "Analyzing" : "Analyze review";
}

function formatLatency(value) {
  const latency = Number(value);
  return Number.isFinite(latency) ? `${latency.toFixed(2)} ms` : "—";
}

function renderModelResult(elementId, result) {
  const card = document.querySelector(elementId);
  const sentiment = String(result?.sentiment || "").toLowerCase();
  card.classList.toggle("is-negative", sentiment === "negative");
  card.querySelector(".sentiment-value").textContent = sentiment || "Unknown";
  card.querySelector(".latency-value").textContent = formatLatency(result?.latency_ms);
}

function renderResults(data) {
  renderModelResult("#result-a", data.model_a);
  renderModelResult("#result-b", data.model_b);

  const agreementBadge = document.querySelector("#agreement-badge");
  agreementBadge.textContent = data.same_prediction ? "✓ Models agree" : "↗ Models differ";
  agreementBadge.style.color = data.same_prediction ? "var(--positive)" : "var(--violet)";
  agreementBadge.style.borderColor = data.same_prediction ? "rgba(79, 214, 160, .25)" : "rgba(156, 121, 255, .3)";
  agreementBadge.style.background = data.same_prediction ? "rgba(79, 214, 160, .08)" : "rgba(156, 121, 255, .08)";

  resultsEmpty.hidden = true;
  resultsContent.hidden = false;
  resultsContent.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("Health check failed");
    setConnectionStatus(true);
  } catch {
    setConnectionStatus(false);
  }
}

async function analyzeReview() {
  const review = reviewInput.value.trim();
  clearError();
  if (!review) {
    showError("Add a movie review before running an analysis.");
    reviewInput.focus();
    return;
  }

  setLoading(true);
  try {
    const response = await fetch(`${API_BASE_URL}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ review }),
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(data?.detail?.[0]?.msg || data?.detail || `API request failed (${response.status})`);
    }
    if (!data?.model_a || !data?.model_b || typeof data.same_prediction !== "boolean") {
      throw new Error("The API returned an unexpected comparison response.");
    }
    setConnectionStatus(true);
    renderResults(data);
  } catch (error) {
    setConnectionStatus(false);
    showError(`Unable to analyze this review. ${error.message || "Check that the API is running and try again."}`);
  } finally {
    setLoading(false);
  }
}

reviewInput.addEventListener("input", updateCharacterCount);
analyzeButton.addEventListener("click", analyzeReview);
reviewInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyzeReview();
});
document.querySelectorAll(".example-btn").forEach((button) => {
  button.addEventListener("click", () => {
    reviewInput.value = button.dataset.review || "";
    updateCharacterCount();
    clearError();
    reviewInput.focus();
  });
});

updateCharacterCount();
checkHealth();
