// llm_ui.js
// Location: C:\Engineers\root\public\console\llm_ui.js

document.addEventListener("DOMContentLoaded", () => {
  const toggleBtn = document.getElementById("toggleBtn");
  const intentSelect = document.getElementById("intent");
  const statusDiv = document.getElementById("modelStatus");

  const fetchStatus = async () => {
    try {
      const res = await fetch("/model_status");
      const json = await res.json();
      if (json.status === "success") {
        statusDiv.textContent = `Active Model: ${json.model || "none"}`;
      } else {
        statusDiv.textContent = `Status error: ${json.message}`;
      }
    } catch (err) {
      statusDiv.textContent = `Fetch error: ${err}`;
    }
  };

  const toggleModel = async () => {
    const intent = intentSelect.value;
    try {
      const res = await fetch("/toggle_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ intent })
      });
      const json = await res.json();
      if (json.status === "success") {
        statusDiv.textContent = json.message;
        fetchStatus();
      } else {
        statusDiv.textContent = `Toggle failed: ${json.message}`;
      }
    } catch (err) {
      statusDiv.textContent = `Error: ${err}`;
    }
  };

  toggleBtn.addEventListener("click", toggleModel);
  fetchStatus();
});
