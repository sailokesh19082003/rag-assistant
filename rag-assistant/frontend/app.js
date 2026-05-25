// ── Session Management ─────────────────────────────────────────────────────
let sessionId = localStorage.getItem("rag_session_id") || generateSessionId();
localStorage.setItem("rag_session_id", sessionId);
document.getElementById("sessionLabel").textContent = "Session: " + sessionId.slice(0, 8) + "…";

function generateSessionId() {
  return "sess_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
}

// ── Health Check ───────────────────────────────────────────────────────────
async function checkHealth() {
  const dot = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.status === "healthy" && data.vectorstore_loaded) {
      dot.className = "status-dot online";
      text.textContent = `${data.total_chunks} chunks indexed`;
    } else {
      dot.className = "status-dot error";
      text.textContent = "Indexing…";
    }
  } catch {
    dot.className = "status-dot error";
    text.textContent = "Offline";
  }
}

// ── Input Handling ─────────────────────────────────────────────────────────
const input = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");

input.addEventListener("input", () => {
  sendBtn.disabled = input.value.trim().length === 0;
});

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

function handleKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
}

function sendSuggestion(text) {
  input.value = text;
  sendBtn.disabled = false;
  sendMessage();
}

// ── Send Message ───────────────────────────────────────────────────────────
let isLoading = false;

async function sendMessage() {
  const message = input.value.trim();
  if (!message || isLoading) return;

  // Hide welcome state on first message
  const welcome = document.getElementById("welcomeState");
  if (welcome) welcome.style.display = "none";

  // Clear input
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;
  isLoading = true;

  // Render user message
  appendMessage("user", message);

  // Show typing indicator
  const typingId = showTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, message }),
    });

    removeTyping(typingId);

    const data = await res.json();

    if (!res.ok) {
      appendError(data.detail || data.error || "An error occurred.");
    } else {
      appendMessage("assistant", data.reply, {
        chunks: data.retrievedChunks,
        tokens: data.tokensUsed,
        scores: data.similarityScores,
      });

      // Update retrieval badge
      const badge = document.getElementById("retrievalBadge");
      const countEl = document.getElementById("retrievalCount");
      if (data.retrievedChunks > 0) {
        badge.style.display = "flex";
        countEl.textContent = `${data.retrievedChunks} chunk${data.retrievedChunks !== 1 ? "s" : ""} retrieved`;
      } else {
        badge.style.display = "none";
      }
    }
  } catch (err) {
    removeTyping(typingId);
    appendError("Network error. Please check your connection.");
    console.error(err);
  } finally {
    isLoading = false;
    input.focus();
  }
}

// ── Render Helpers ─────────────────────────────────────────────────────────
function appendMessage(role, text, meta = null) {
  const container = document.getElementById("messagesContainer");

  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  // Simple markdown-ish rendering
  bubble.innerHTML = renderMarkdown(text);

  row.appendChild(bubble);

  // Metadata row
  if (meta) {
    if (meta.chunks > 0) {
      const chunkPill = document.createElement("div");
      chunkPill.className = "chunk-info";
      chunkPill.innerHTML = `
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        ${meta.chunks} source${meta.chunks !== 1 ? "s" : ""} retrieved
        ${meta.scores && meta.scores.length ? "· top score: " + meta.scores[0].toFixed(3) : ""}
      `;
      row.insertBefore(chunkPill, bubble);
    }
  }

  const metaDiv = document.createElement("div");
  metaDiv.className = "message-meta";
  metaDiv.textContent = formatTime(new Date());
  if (role === "assistant" && meta && meta.tokens) {
    metaDiv.textContent += ` · ${meta.tokens} tokens`;
  }
  row.appendChild(metaDiv);

  container.appendChild(row);
  scrollToBottom();
}

function appendError(msg) {
  const container = document.getElementById("messagesContainer");
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.innerHTML = `<div class="message-error">⚠ ${escapeHtml(msg)}</div>`;
  container.appendChild(row);
  scrollToBottom();
}

function showTyping() {
  const id = "typing_" + Date.now();
  const container = document.getElementById("messagesContainer");
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.id = id;
  row.innerHTML = `
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>`;
  container.appendChild(row);
  scrollToBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollToBottom() {
  const c = document.getElementById("messagesContainer");
  c.scrollTop = c.scrollHeight;
}

// ── New Chat ────────────────────────────────────────────────────────────────
document.getElementById("newChatBtn").addEventListener("click", async () => {
  // Clear server-side history
  try {
    await fetch(`/api/session/${sessionId}`, { method: "DELETE" });
  } catch {}

  // New session
  sessionId = generateSessionId();
  localStorage.setItem("rag_session_id", sessionId);
  document.getElementById("sessionLabel").textContent = "Session: " + sessionId.slice(0, 8) + "…";

  // Clear UI
  const container = document.getElementById("messagesContainer");
  container.innerHTML = `
    <div class="welcome-state" id="welcomeState">
      <div class="welcome-icon">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </div>
      <h2>Ask me anything</h2>
      <p>I'll search the knowledge base to give you accurate, grounded answers.</p>
      <div class="suggestion-chips">
        <button class="chip" onclick="sendSuggestion('How do I reset my password?')">How do I reset my password?</button>
        <button class="chip" onclick="sendSuggestion('What subscription plans are available?')">What subscription plans are available?</button>
        <button class="chip" onclick="sendSuggestion('How do I delete my account?')">How do I delete my account?</button>
        <button class="chip" onclick="sendSuggestion('How do I contact support?')">How do I contact support?</button>
      </div>
    </div>`;

  document.getElementById("retrievalBadge").style.display = "none";
  input.focus();
});

// ── Utilities ─────────────────────────────────────────────────────────────
function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`(.*?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

// ── Init ───────────────────────────────────────────────────────────────────
checkHealth();
setInterval(checkHealth, 30000);
input.focus();
