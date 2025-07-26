
function loginUser(event) {
  event.preventDefault();
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value.trim();
  const error = document.getElementById("loginError");

 
  if (username === "farmer" && password === "12345") {
    // Redirect to home page on success
    window.location.href = "{{ url_for('home') }}".replace('/login', '/home');
  } else {
    error.textContent = "Invalid credentials!";
  }
}


document.addEventListener("DOMContentLoaded", () => {

  // Fertilizer recommendation form handler
  const fertilizerForm = document.getElementById("fertilizerForm");
  if (fertilizerForm) {
    fertilizerForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const data = {
        Temperature: parseFloat(document.getElementById("Temperature").value),
        Humidity: parseFloat(document.getElementById("Humidity").value),
        Moisture: parseFloat(document.getElementById("Moisture").value),
        Nitrogen: parseFloat(document.getElementById("Nitrogen").value),
        Phosphorus: parseFloat(document.getElementById("Phosphorus").value),
        Potassium: parseFloat(document.getElementById("Potassium").value),
        "Soil Type": document.getElementById("Soil Type").value,
        "Crop Type": document.getElementById("Crop Type").value,
      };

      const resultBox = document.getElementById("result");
      resultBox.style.display = "block";
      resultBox.innerHTML = "Processing...";

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });

        if (!response.ok) {
          const errData = await response.json();
          resultBox.innerHTML = `<span class="error-text">Error: ${errData.error || response.statusText}</span>`;
          return;
        }
        const result = await response.json();

        if (result.recommendation) {
          resultBox.innerHTML = `
            <b>Recommended Fertilizer:</b> ${result.recommendation}<br>
            <b>Confidence:</b> ${result.confidence}%
          `;
        } else {
          resultBox.innerHTML = `<span class="error-text">Error: ${result.error || "Unknown error"}</span>`;
        }
      } catch (error) {
        resultBox.innerHTML = `<span class="error-text">Could not connect to backend. Make sure Flask is running on port 5000.</span>`;
        console.error("Error:", error);
      }
    });
  }
});




