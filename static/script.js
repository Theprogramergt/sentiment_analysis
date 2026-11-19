
const reviewInput = document.getElementById("review-input");
const charCount = document.getElementById("char-count");
const analyzeBtn = document.getElementById("analyze-btn");
const errorMessage = document.getElementById("error-message");

const connectionStatus = document.getElementById("connection-status");
const resultsEmpty = document.getElementById("results-empty");
const resultsContent = document.getElementById("results-content");
const MODEL_ACCURACY = {
  "result-a": 85.05,
  "result-b": 88.36,
};

const positiveExample =
  "A beautifully acted film with a thoughtful story and an ending that stayed with me.";

const negativeExample =
  "The plot was predictable, the pacing dragged, and the characters felt completely forgettable.";

// Update character counter
reviewInput.addEventListener("input", () => {
  charCount.textContent = `${reviewInput.value.length} / 2,000`;
});

// Example buttons
document.querySelectorAll(".example-btn").forEach((button) => {
  button.addEventListener("click", () => {
    reviewInput.value = button.dataset.review;
    reviewInput.dispatchEvent(new Event("input"));
    errorMessage.hidden = true;
    reviewInput.focus();
  });
});

// Check API connection
async function checkAPI() {
  try {
    const response = await fetch("/health");

    if (!response.ok) {
      throw new Error("API unavailable");
    }

    connectionStatus.classList.add("is-online");
    connectionStatus.classList.remove("is-offline");
    connectionStatus.querySelector(".connection-label").textContent =
      "API Connected";
  } catch (error) {
    connectionStatus.classList.add("is-offline");
    connectionStatus.classList.remove("is-online");
    connectionStatus.querySelector(".connection-label").textContent =
      "API Offline";
  }
}

checkAPI();

const abDatasetNote = document.getElementById("ab-dataset-note");
const startAbBtn = document.getElementById("start-ab-btn");
const abErrorMessage = document.getElementById("ab-error-message");
const abResults = document.getElementById("ab-results");

async function loadAbDataset() {
  try {
    const response = await fetch("/ab-test/dataset");
    if (!response.ok) {
      throw new Error("Dataset unavailable");
    }
    const data = await response.json();
    abDatasetNote.textContent =
      `${data.review_count} held-out IMDb reviews`;
  } catch (error) {
    abDatasetNote.textContent = "Held-out IMDb dataset unavailable";
  }
}

function fillAbModel(prefix, model) {
  document.getElementById(`${prefix}-accuracy`).textContent =
    `${model.accuracy_percent.toFixed(2)}%`;
  document.getElementById(`${prefix}-processed`).textContent =
    model.processed;
  document.getElementById(`${prefix}-correct`).textContent =
    model.correct;
  document.getElementById(`${prefix}-average`).textContent =
    `${model.average_inference_ms.toFixed(2)} ms`;
  document.getElementById(`${prefix}-total`).textContent =
    `${model.total_inference_ms.toFixed(2)} ms`;
}

function displayTrafficDistribution(traffic) {
  document.getElementById("traffic-total").textContent = traffic.total;
  document.getElementById("traffic-a-count").textContent =
    traffic.model_a.count;
  document.getElementById("traffic-b-count").textContent =
    traffic.model_b.count;
  document.getElementById("traffic-a-percent").textContent =
    `${traffic.model_a.percentage.toFixed(1)}%`;
  document.getElementById("traffic-b-percent").textContent =
    `${traffic.model_b.percentage.toFixed(1)}%`;
  document.getElementById("traffic-bar-a").style.width =
    `${traffic.model_a.percentage}%`;
  document.getElementById("traffic-bar-b").style.width =
    `${traffic.model_b.percentage}%`;
}

function displaySampleAssignments(samples) {
  const tableBody = document.getElementById("assignment-table-body");
  tableBody.replaceChildren();

  samples.slice(0, 8).forEach((sample) => {
    const row = document.createElement("tr");

    const idCell = document.createElement("td");
    idCell.textContent = `#${sample.review_id}`;

    const previewCell = document.createElement("td");
    previewCell.className = "assignment-review-preview";
    previewCell.textContent =
      sample.preview.length >= 160
        ? `${sample.preview}...`
        : sample.preview;

    const modelCell = document.createElement("td");
    const modelBadge = document.createElement("span");
    modelBadge.className =
      `assignment-model-badge ${sample.assigned_model === "A" ? "model-a" : "model-b"}`;
    modelBadge.textContent = `Model ${sample.assigned_model}`;
    modelCell.appendChild(modelBadge);

    const sentimentCell = document.createElement("td");
    sentimentCell.className = "assignment-sentiment";
    sentimentCell.textContent = sample.actual_sentiment;

    row.append(idCell, previewCell, modelCell, sentimentCell);
    tableBody.appendChild(row);
  });
}

function displayAbResults(data) {
  fillAbModel("ab-a", data.model_a);
  fillAbModel("ab-b", data.model_b);
  abDatasetNote.textContent =
    `${data.review_count} held-out IMDb reviews / seed ${data.seed}`;
  displayTrafficDistribution(data.traffic);
  displaySampleAssignments(data.sample_assignments || []);

  const significance = data.significance;
  document.getElementById("ab-conclusion").textContent =
    significance.conclusion;
  document.getElementById("ab-recommendation").textContent =
    significance.recommendation;
  document.getElementById("ab-z").textContent =
    significance.z_statistic.toFixed(4);
  document.getElementById("ab-p").textContent =
    significance.p_value.toFixed(6);
  document.getElementById("ab-threshold").textContent =
    `±${significance.significance_threshold.toFixed(2)}`;
  abResults.hidden = false;
  abResults.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

loadAbDataset();

async function retrieveAbResults() {
  try {
    const response = await fetch("/ab-test/results");
    if (!response.ok) return;
    const data = await response.json();
    if (data.status === "completed") {
      displayAbResults(data);
    }
  } catch (error) {
    console.error("Unable to retrieve A/B results:", error);
  }
}

retrieveAbResults();

startAbBtn.addEventListener("click", async () => {
  abErrorMessage.hidden = true;
  startAbBtn.disabled = true;
  startAbBtn.querySelector(".button-label").textContent =
    "Running experiment...";

  try {
    const response = await fetch("/ab-test/start", { method: "POST" });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail || `Server error: ${response.status}`);
    }
    const data = await response.json();
    displayAbResults(data);
  } catch (error) {
    abErrorMessage.textContent =
      "Unable to run the A/B test. " + error.message;
    abErrorMessage.hidden = false;
  } finally {
    startAbBtn.disabled = false;
    startAbBtn.querySelector(".button-label").textContent =
      "Start A/B Test";
  }
});

// Display model results
function displayResult(cardId, result) {
  const card = document.getElementById(cardId);
  const sentiment = result.sentiment.toLowerCase();

  card.classList.remove("positive", "negative");
  card.classList.add(sentiment);

  card.querySelector(".sentiment-value").textContent =
    sentiment.charAt(0).toUpperCase() + sentiment.slice(1);

  card.querySelector(".sentiment-icon").textContent =
    sentiment === "positive" ? "✦" : "−";

  card.querySelector(".latency-value").textContent =
    `${result.latency_ms} ms`;

  const accuracy = Number(
    result.accuracy ??
      result.accuracy_percent ??
      MODEL_ACCURACY[cardId],
  );
  card.querySelector(".accuracy-value").textContent = Number.isFinite(accuracy)
    ? `${accuracy > 1 ? accuracy : accuracy * 100}%`
    : "Not available";
}

// Analyze review
analyzeBtn.addEventListener("click", async () => {
  const review = reviewInput.value.trim();

  errorMessage.hidden = true;

  if (!review) {
    errorMessage.textContent = "Please enter a movie review first.";
    errorMessage.hidden = false;
    reviewInput.focus();
    return;
  }

  const buttonLabel = analyzeBtn.querySelector(".button-label");

  analyzeBtn.disabled = true;
  buttonLabel.textContent = "Analyzing...";

  try {
    const response = await fetch("/compare", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ review }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    displayResult("result-a", data.model_a);
    displayResult("result-b", data.model_b);

    const agreementBadge = document.getElementById("agreement-badge");

    agreementBadge.textContent = data.same_prediction
      ? "Models agree"
      : "Models disagree";

    resultsEmpty.hidden = true;
    resultsContent.hidden = false;

    resultsContent.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  } catch (error) {
    errorMessage.textContent =
      "Unable to analyze the review. Please check your connection and try again.";

    errorMessage.hidden = false;
    console.error("Analysis error:", error);
  } finally {
    analyzeBtn.disabled = false;
    buttonLabel.textContent = "Analyze review";
  }
});
