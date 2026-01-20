// nexus_console.js - full logic backend for the console UI

const themes = {
  water: { folder: "videos/water/", files: ["Water1.mp4", "Water2.mp4"] },
  northern: { folder: "videos/northern/", files: ["Aurora1.mp4", "Aurora2.mp4"] },
  smoke: { folder: "videos/smoke/", files: ["Smoke.mp4", "Smoke1.mp4"] }
};

let currentTheme = "water";
const videoElement = document.getElementById("backgroundVideo");
function switchTheme(themeKey) {
  currentTheme = themeKey;
  const theme = themes[themeKey];
  const file = theme.files[Math.floor(Math.random() * theme.files.length)];
  videoElement.src = theme.folder + file;
  videoElement.load();
  videoElement.play();
}
switchTheme("water");

let recognizing = false;
let recognition;
document.getElementById("micBtn").onclick = () => {
  if (!('webkitSpeechRecognition' in window)) return alert("Speech-to-text not supported.");
  if (!recognition) {
    recognition = new webkitSpeechRecognition();
    recognition.lang = 'en-US';
    recognition.onresult = e => {
      document.getElementById('userInput').value = e.results[0][0].transcript;
    };
  }
  if (!recognizing) {
    recognizing = true;
    switchTheme("northern");
    recognition.start();
  } else {
    recognition.stop();
    recognizing = false;
    switchTheme("water");
  }
};

let agentThreads = {};
let chatHistory = [];
const chatBox = document.getElementById("chatHistory");
function renderChat() {
  chatBox.innerHTML = chatHistory.map(m =>
    `<div class="chat-message ${m.sender === 'You' ? 'user' : 'agent'}"><b>${m.sender}:</b> ${m.text}</div>`
  ).join('');
  chatBox.scrollTop = chatBox.scrollHeight;
}

document.getElementById("inputForm").onsubmit = async (e) => {
  e.preventDefault();
  const input = document.getElementById("userInput").value.trim();
  if (!input) return;
  document.getElementById("userInput").value = '';
  chatHistory.push({ sender: "You", text: input });
  renderChat();
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input })
    });
    const data = await res.json();
    const reply = data.response || data.error || "[No reply]";
    chatHistory.push({ sender: "Engineers", text: reply });
    renderChat();
    speak(reply);
  } catch (err) {
    chatHistory.push({ sender: "System", text: "❌ Error sending message." });
    renderChat();
  }
};

function speak(text) {
  fetch("/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  })
    .then(res => res.blob())
    .then(blob => {
      const audio = new Audio(URL.createObjectURL(blob));
      audio.play();
    });
}

async function loadModels() {
  const list = document.getElementById("modelList");
  list.innerHTML = "<li>Loading models...</li>";
  try {
    const res = await fetch("/loaded_models");
    const models = await res.json();
    agentThreads = {};
    list.innerHTML = "";
    models.forEach(model => {
      agentThreads[model] = [];
      const li = document.createElement("li");
      li.innerHTML = `✅ <span style="font-size: 0.85rem;">${model}</span>`;
      list.appendChild(li);
    });
    document.getElementById("connectionStatus").innerText = "✅ Models Loaded from LM Studio";
  } catch (err) {
    document.getElementById("connectionStatus").innerText = "⚠️ Could not load models";
    list.innerHTML = "<li>Could not load models.</li>";
  }
}

function runModelSelector() {
  fetch("/run_model_selector")
    .then(res => res.json())
    .then(data => {
      if (data.status === "launched") alert("✅ Model selector launched.");
      else alert("❌ " + (data.error || "unknown"));
    })
    .catch(err => alert("❌ Error: " + err.message));
}

document.getElementById("fileUpload").addEventListener("change", async function () {
  const files = this.files;
  const formData = new FormData();
  for (let file of files) formData.append("files", file);
  const res = await fetch("/upload", { method: "POST", body: formData });
  const data = await res.json();
  document.getElementById("uploadStatus").innerText = `✅ Uploaded: ${data.saved.join(', ')}`;
  loadFiles();
});

async function loadFiles() {
  const list = document.getElementById("fileList");
  const res = await fetch("/files");
  const files = await res.json();
  list.innerHTML = files.map(name => `<li>${name} <button onclick="deleteFile('${name}')">🗑️</button></li>`).join('');
}
async function deleteFile(name) {
  await fetch(`/delete/${name}`, { method: "DELETE" });
  loadFiles();
}

function loadLogs() {
  document.getElementById("logViewer").textContent = "[Log stream placeholder — connect backend to read logs]";
}

async function updateDiagnostics() {
  try {
    const res = await fetch("/diagnostics");
    const data = await res.json();
    const status = `LM Studio: ${data.lmstudio ? '✅' : '❌'} | Models: ${data.models_loaded} | Status: ${data.status}`;
    document.getElementById("diagnosticsStatus").innerText = status;
  } catch (err) {
    document.getElementById("diagnosticsStatus").innerText = "❌ Diagnostics unavailable";
  }
}

async function updateSystemStats() {
  try {
    const res = await fetch("/system_stats");
    const stats = await res.json();
    document.getElementById("systemStats").innerText = `CPU: ${stats.cpu}% | RAM: ${stats.memory}% | GPU: ${stats.gpu || 'N/A'}`;
  } catch (err) {
    document.getElementById("systemStats").innerText = "⚠️ System stats unavailable.";
  }
}

async function loadRightBar() {
  try {
    const res = await fetch("/loaded_models");
    const models = await res.json();
    const llmContainer = document.getElementById("llmChatCards");
    llmContainer.innerHTML = "";
    models.forEach(modelId => {
      const card = document.createElement("div");
      card.className = "llm-card";
      card.innerHTML = `
        <div><b>${modelId}</b></div>
        <input type="text" placeholder="Ask ${modelId}..." onkeydown="if(event.key==='Enter')sendToModel('${modelId}', this)">
        <div id="response-${modelId}" style="margin-top: 0.5rem; opacity: 0.8;"></div>
      `;
      llmContainer.appendChild(card);
    });
  } catch {
    document.getElementById("llmChatCards").innerHTML = "<i>Failed to load models</i>";
  }
}

async function sendToModel(model, inputElem) {
  const msg = inputElem.value.trim();
  if (!msg) return;
  inputElem.value = "Sending...";
  try {
    const res = await fetch(`/single_chat/${model}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg })
    });
    const data = await res.json();
    document.getElementById("response-" + model).innerText = data.response || data.error || "[No reply]";
  } catch {
    document.getElementById("response-" + model).innerText = "❌ Failed to reach model.";
  } finally {
    inputElem.value = "";
  }
}

function autoRefresh() {
  updateDiagnostics();
  updateSystemStats();
  loadModels();
  loadRightBar();
  loadFiles();
  loadLogs();
}

// Startup
autoRefresh();
setInterval(() => {
  updateSystemStats();
  updateDiagnostics();
}, 10000);

document.getElementById("taskList").addEventListener("input", e =>
  localStorage.setItem("taskList", e.target.value));
document.getElementById("projectNotes").addEventListener("input", e =>
  localStorage.setItem("projectNotes", e.target.value));
document.getElementById("taskList").value = localStorage.getItem("taskList") || "";
document.getElementById("projectNotes").value = localStorage.getItem("projectNotes") || "";
