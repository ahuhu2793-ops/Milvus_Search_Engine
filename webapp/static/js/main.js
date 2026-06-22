/* ============================================================
   EduSearch – main.js  (with Fault Tolerance real-time panel)
   ============================================================ */

// ── State ─────────────────────────────────────────────────
let currentCategory = "All";
let currentQuery    = "";
let searchTimeout   = null;
let isSearching     = false;
let healthTimer     = null;

// ── DOM refs ──────────────────────────────────────────────
const searchInput  = document.getElementById("search-input");
const searchBtn    = document.getElementById("search-btn");
const resultsGrid  = document.getElementById("results-grid");
const loadingEl    = document.getElementById("loading");
const emptyState   = document.getElementById("empty-state");
const noResults    = document.getElementById("no-results");
const resultsMeta  = document.getElementById("results-meta");
const resultCount  = document.getElementById("result-count");
const resultQuery  = document.getElementById("result-query");
const nodeBadgesRow= document.getElementById("node-badges-row");
const filterBtns   = document.querySelectorAll(".filter-btn");
const statNumbers  = document.querySelectorAll(".stat-number");
const ftBanner     = document.getElementById("ft-banner");
const ftBannerMsg  = document.getElementById("ft-banner-msg");

const CAT_CLASS = {
  "Artificial Intelligence": "cat-ai",
  "Machine Learning":        "cat-ml",
  "Deep Learning":           "cat-dl",
  "NLP":                     "cat-nlp",
  "Computer Vision":         "cat-cv",
};

// ── Counter animation ──────────────────────────────────────
function animateCounter(el) {
  const target = parseInt(el.dataset.count, 10);
  let current  = 0;
  const step   = Math.ceil(target / 40);
  const timer  = setInterval(() => {
    current += step;
    if (current >= target) { current = target; clearInterval(timer); }
    el.textContent = current;
  }, 30);
}

// ── Node status health check ───────────────────────────────
async function fetchNodeStatus() {
  try {
    const res  = await fetch("/api/node-status");
    const data = await res.json();
    renderNodeStatus(data.nodes);
  } catch (e) {
    console.warn("Node status fetch failed:", e);
  }
}

function renderNodeStatus(nodes) {
  nodes.forEach((node, idx) => {
    const num       = idx + 1;
    const badge     = document.getElementById(`node${num}-badge`);
    const latencyEl = document.getElementById(`node${num}-latency`);
    const faultTag  = document.getElementById(`node${num}-fault`);
    const card      = document.getElementById(`node-card-${num}`);

    if (!badge) return;

    const isOnline = node.status === "online";

    // Badge
    badge.className = `node-status-badge ${isOnline ? "online" : "offline"}`;
    badge.querySelector(".status-text").textContent =
      isOnline ? "● Online" : "✕ Offline";

    // Card highlight
    card.classList.toggle("online",  isOnline);
    card.classList.toggle("offline", !isOnline);

    // Latency
    latencyEl.textContent = isOnline ? `${node.latency_ms} ms` : "– ms";

    // Fault tag
    faultTag.style.display = isOnline ? "none" : "block";
  });

  // Flow lines in coordinator
  nodes.forEach((node, idx) => {
    const lineId = idx === 0 ? "flow-node1" : "flow-node2";
    const line   = document.getElementById(lineId);
    if (!line) return;
    line.classList.toggle("dead", node.status !== "online");
  });

  // FT Banner
  const anyOffline = nodes.some(n => n.status !== "online");
  const allOffline = nodes.every(n => n.status !== "online");

  if (allOffline) {
    ftBanner.style.display = "flex";
    ftBannerMsg.textContent =
      "⚠️ Cả 2 node đều offline! Hệ thống dùng Local AI Search làm fallback.";
  } else if (anyOffline) {
    ftBanner.style.display = "flex";
    const deadNode = nodes.find(n => n.status !== "online");
    ftBannerMsg.textContent =
      `${deadNode.name} bị tắt — Coordinator bắt exception, trả [] cho node đó và vẫn trả kết quả từ node còn lại. Hệ thống không crash!`;
  } else {
    ftBanner.style.display = "none";
  }
}

// ── Init ──────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Counters
  setTimeout(() => statNumbers.forEach(animateCounter), 600);

  // Node health check every 5 s
  fetchNodeStatus();
  healthTimer = setInterval(fetchNodeStatus, 5000);

  // Filter buttons
  filterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      filterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentCategory = btn.dataset.cat;
      if (currentQuery) doSearch(currentQuery);
    });
  });

  // Quick tags
  document.querySelectorAll(".qtag").forEach(tag => {
    tag.addEventListener("click", () => {
      const q = tag.dataset.q;
      searchInput.value = q;
      doSearch(q);
    });
  });

  // Featured cards
  document.querySelectorAll(".featured-card").forEach(card => {
    card.addEventListener("click", () => {
      const q = card.dataset.q;
      searchInput.value = q;
      doSearch(q);
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });

  // Search btn
  searchBtn.addEventListener("click", () => {
    const q = searchInput.value.trim();
    if (q) doSearch(q);
  });

  // Enter key
  searchInput.addEventListener("keydown", e => {
    if (e.key === "Enter") {
      const q = searchInput.value.trim();
      if (q) doSearch(q);
    }
  });

  // Live debounce
  searchInput.addEventListener("input", () => {
    clearTimeout(searchTimeout);
    const q = searchInput.value.trim();
    if (!q) { showEmpty(); return; }
    searchTimeout = setTimeout(() => doSearch(q), 450);
  });
});

// ── Search ─────────────────────────────────────────────────
async function doSearch(query) {
  if (isSearching) return;
  currentQuery = query;
  isSearching  = true;
  showLoading();

  try {
    const params = new URLSearchParams({ q: query, category: currentCategory });
    const res    = await fetch(`/api/search?${params}`);
    const data   = await res.json();
    renderResults(data);

    // Update per-node result counts in the node panel
    if (data.node_details) {
      updateNodeResultCounts(data.node_details);
    }

    // Re-fetch health immediately after a search (latency updated)
    fetchNodeStatus();

  } catch (err) {
    console.error("Search error:", err);
    showError();
  } finally {
    isSearching = false;
  }
}

// Update last-query result count on each node card
function updateNodeResultCounts(nodeDetails) {
  const entries = Object.entries(nodeDetails);
  entries.forEach(([name, detail], idx) => {
    const num = name.includes("1") ? 1 : 2;
    const el  = document.getElementById(`node${num}-result-val`);
    if (!el) return;
    if (detail.ok) {
      el.textContent = `${detail.count} kết quả · ${detail.latency_ms}ms`;
      el.style.color = "#34d399";
    } else {
      el.textContent = "Node offline — không có kết quả";
      el.style.color = "#f87171";
    }

    // Update latency from search response
    const latEl = document.getElementById(`node${num}-latency`);
    if (latEl) {
      latEl.textContent = detail.ok ? `${detail.latency_ms} ms` : "– ms";
    }
  });
}

// ── Render results ─────────────────────────────────────────
function renderResults(data) {
  hideAll();

  if (!data.results || data.results.length === 0) {
    noResults.style.display = "block";
    return;
  }

  // Meta bar
  resultsMeta.style.display = "flex";
  resultCount.textContent   = `Tìm thấy ${data.total} tài liệu`;
  resultQuery.textContent   = `cho "${data.query}"`;

  // Node badges in meta bar
  nodeBadgesRow.innerHTML = "";
  if (data.node_details) {
    Object.entries(data.node_details).forEach(([name, detail]) => {
      const badge = document.createElement("span");
      badge.className = `node-result-badge ${detail.ok ? "ok" : "failed"}`;
      badge.textContent = detail.ok
        ? `✓ ${name}: ${detail.count} docs`
        : `✕ ${name}: offline`;
      nodeBadgesRow.appendChild(badge);
    });
  }

  if (data.fallback) {
    const tag = document.createElement("span");
    tag.className = "node-result-badge";
    tag.style.background = "rgba(245,158,11,.12)";
    tag.style.border     = "1px solid rgba(245,158,11,.3)";
    tag.style.color      = "#fbbf24";
    tag.textContent      = "⚡ Local Fallback";
    nodeBadgesRow.appendChild(tag);
  }

  // Cards
  resultsGrid.innerHTML = "";
  data.results.forEach((doc, i) => {
    const card = buildCard(doc, i);
    resultsGrid.appendChild(card);
  });

  // Animate bars
  requestAnimationFrame(() => {
    document.querySelectorAll(".relevance-fill").forEach(bar => {
      bar.style.width = bar.dataset.width + "%";
    });
  });
}

function buildCard(doc, index) {
  const catClass = CAT_CLASS[doc.category] || "cat-ml";
  const score    = Math.min(doc.score, 100);
  const deg      = Math.round((score / 100) * 360);
  const nodeSrc  = doc.source_node || "";
  const nodeClass = nodeSrc.includes("1") ? "node1" : nodeSrc.includes("2") ? "node2" : "local";
  const nodeLabel = nodeSrc || "Local";

  const card = document.createElement("div");
  card.className = "result-card";
  card.style.animationDelay = `${index * 0.06}s`;

  card.innerHTML = `
    <div class="card-node-tag ${nodeClass}">
      🖥️ ${escapeHtml(nodeLabel)}
    </div>
    <div class="card-top">
      <span class="card-category-badge ${catClass}">
        ${doc.icon} ${doc.category}
      </span>
      <div class="score-circle" style="--deg:${deg}deg">
        <span class="score-value">${score.toFixed(0)}</span>
        <span class="score-pct">%</span>
      </div>
    </div>
    <h2 class="card-title">${escapeHtml(doc.title)}</h2>
    <div class="relevance-bar-wrap">
      <div class="relevance-label">
        <span>Độ phù hợp</span>
        <span>${score.toFixed(1)}%</span>
      </div>
      <div class="relevance-bar">
        <div class="relevance-fill" data-width="${score}" style="width:0%"></div>
      </div>
    </div>
    <p class="card-desc">${escapeHtml(doc.description)}</p>
    <div class="card-footer">
      <div class="card-meta">
        <span class="meta-item">✍️ ${escapeHtml(doc.author)}</span>
        <span class="meta-item">📅 ${doc.year}</span>
        <span class="meta-item">📄 ${doc.pages} trang</span>
      </div>
      <button class="card-read-btn" onclick="handleRead(event,'${escapeHtml(doc.title)}')">
        Xem chi tiết →
      </button>
    </div>
  `;
  return card;
}

function handleRead(e, title) {
  e.stopPropagation();
  showToast(`📖 Đang mở: ${title}`);
}

// ── Visibility helpers ─────────────────────────────────────
function hideAll() {
  loadingEl.style.display   = "none";
  emptyState.style.display  = "none";
  noResults.style.display   = "none";
  resultsMeta.style.display = "none";
  resultsGrid.innerHTML     = "";
}
function showLoading() { hideAll(); loadingEl.style.display = "flex"; }
function showEmpty()   { hideAll(); emptyState.style.display = "block"; }
function showError()   { hideAll(); noResults.style.display  = "block"; }

// ── Toast ──────────────────────────────────────────────────
function showToast(message) {
  const old = document.getElementById("toast-container");
  if (old) old.remove();
  const toast = document.createElement("div");
  toast.id = "toast-container";
  toast.style.cssText = `
    position:fixed;bottom:28px;right:28px;
    background:rgba(15,20,40,.95);backdrop-filter:blur(20px);
    border:1px solid rgba(99,102,241,.35);color:#e2e8f0;
    padding:14px 22px;border-radius:14px;
    font-family:'Inter',sans-serif;font-size:.9rem;font-weight:600;
    z-index:9999;box-shadow:0 16px 40px rgba(0,0,0,.5);
    transform:translateY(20px);opacity:0;
    transition:all .35s cubic-bezier(.34,1.56,.64,1);
  `;
  toast.textContent = message;
  document.body.appendChild(toast);
  requestAnimationFrame(() => { toast.style.transform="translateY(0)"; toast.style.opacity="1"; });
  setTimeout(() => {
    toast.style.transform="translateY(20px)"; toast.style.opacity="0";
    setTimeout(() => toast.remove(), 400);
  }, 3000);
}

// ── Escape ─────────────────────────────────────────────────
function escapeHtml(str) {
  if (!str) return "";
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
