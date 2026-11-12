
const reviewInput = document.getElementById("review-input");
const charCount = document.getElementById("char-count");
const analyzeBtn = document.getElementById("analyze-btn");
const errorMessage = document.getElementById("error-message");

const connectionStatus = document.getElementById("connection-status");
const resultsEmpty = document.getElementById("results-empty");
const resultsContent = document.getElementById("results-content");

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

    connectionStatus.classList.add("connected");
    connectionStatus.classList.remove("disconnected");
    connectionStatus.querySelector(".connection-label").textContent =
      "API Connected";
  } catch (error) {
    connectionStatus.classList.add("disconnected");
    connectionStatus.classList.remove("connected");
    connectionStatus.querySelector(".connection-label").textContent =
      "API Offline";
  }
}

checkAPI();

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
