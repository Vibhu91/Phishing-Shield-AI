/**
 * app.js
 * ======
 * Frontend Controller for Phishing Shield AI Dashboard.
 * Handles tab navigation, sample preloading, API requests, and SVG gauge animation.
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  loadSamplePresets();
  initEmailForm();
  initUrlForm();
  loadBenchmarks();
});

// -----------------------------------------------------------------------------
// 1. Tab Navigation
// -----------------------------------------------------------------------------
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabBtns.forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");

      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });
}

// -----------------------------------------------------------------------------
// 2. Preload Samples
// -----------------------------------------------------------------------------
let samplePresets = [];

async function loadSamplePresets() {
  try {
    const res = await fetch("/api/samples");
    if (!res.ok) return;
    samplePresets = await res.json();

    renderSamplePills();
  } catch (err) {
    console.warn("Could not load sample presets:", err);
  }
}

function renderSamplePills() {
  const emailContainer = document.getElementById("email-sample-pills");
  const urlContainer = document.getElementById("url-sample-pills");

  if (!emailContainer || !urlContainer) return;

  emailContainer.innerHTML = "";
  urlContainer.innerHTML = "";

  samplePresets.forEach(sample => {
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = "sample-pill";
    pill.textContent = sample.title;

    if (sample.type === "email") {
      pill.addEventListener("click", () => {
        document.getElementById("email-subject").value = sample.subject;
        document.getElementById("email-body").value = sample.body;
      });
      emailContainer.appendChild(pill);
    } else if (sample.type === "url") {
      pill.addEventListener("click", () => {
        document.getElementById("url-input").value = sample.url;
      });
      urlContainer.appendChild(pill);
    }
  });
}

// -----------------------------------------------------------------------------
// 3. Email Form & Inference Handler
// -----------------------------------------------------------------------------
function initEmailForm() {
  const form = document.getElementById("email-form");
  const submitBtn = document.getElementById("btn-scan-email");
  const clearBtn = document.getElementById("btn-clear-email");
  const resultBox = document.getElementById("email-result-box");

  clearBtn.addEventListener("click", () => {
    document.getElementById("email-subject").value = "";
    document.getElementById("email-body").value = "";
    resetResultBox(resultBox, "Enter an email on the left or select a quick test preset to run live inference.");
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const subject = document.getElementById("email-subject").value.trim();
    const body = document.getElementById("email-body").value.trim();

    if (!subject && !body) return;

    // Loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span class="spinner"></span> <span>Running Dual-Engine Inference...</span>`;

    resultBox.innerHTML = `
      <div class="empty-state">
        <span class="spinner" style="width: 32px; height: 32px; border-width: 3px; margin-bottom: 14px;"></span>
        <p style="color: var(--accent-cyan); font-weight: 500;">
          Analyzing semantic intent via DistilBERT &amp; scanning embedded links via URL DNN...
        </p>
      </div>
    `;

    try {
      const res = await fetch("/api/predict/email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject, body })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Inference failed.");
      }

      renderEmailResult(resultBox, data);
    } catch (err) {
      resultBox.innerHTML = `
        <div class="verdict-banner phish" style="border-color: var(--border-danger);">
          <div class="verdict-text-group">
            <h3>Inference Error</h3>
            <p style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">${err.message}</p>
          </div>
        </div>
      `;
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span class="btn-text">Analyze Email Threat</span>`;
    }
  });
}

function renderEmailResult(container, data) {
  const isPhish = data.verdict === "PHISHING";
  const prob = data.combined_phishing_prob;
  const pct = Math.round(prob * 100);

  let bannerClass = isPhish ? "phish" : "safe";
  let statusText = isPhish ? "🚨 Phishing Threat Detected" : "🛡️ Legitimate Message";
  let badgeColor = isPhish ? "var(--accent-rose)" : "var(--accent-emerald)";

  // Sub-scores
  const emailNLPScore = Math.round((data.email_text_phishing_prob || 0) * 100);
  const maxUrlScore = data.max_url_phishing_prob !== null ? Math.round(data.max_url_phishing_prob * 100) : null;

  // Build Links Cards
  let linksHtml = "";
  if (data.urls_analysis && data.urls_analysis.length > 0) {
    const cards = data.urls_analysis.map(u => {
      const uIsPhish = u.verdict === "PHISHING";
      const uProb = Math.round(u.phishing_prob * 100);
      const cardClass = uIsPhish ? "link-card link-danger" : "link-card";
      const pillClass = uIsPhish ? "link-risk-pill danger" : "link-risk-pill safe";

      let signalsHtml = "";
      if (u.key_signals && u.key_signals.length > 0) {
        signalsHtml = `<div class="signals-tag-list">${u.key_signals.map(s => `<span class="signal-tag">${s}</span>`).join("")}</div>`;
      }

      return `
        <div class="${cardClass}">
          <div class="link-top-row">
            <span class="link-url-text mono">${escapeHtml(u.url)}</span>
            <span class="${pillClass}">${uProb}% Risk (${u.risk_level})</span>
          </div>
          ${signalsHtml}
        </div>
      `;
    }).join("");

    linksHtml = `
      <div class="links-section">
        <div class="links-header">
          <span>Embedded Hyperlinks Inspected (${data.urls_detected_count})</span>
          <span style="font-size: 0.75rem; color: var(--accent-cyan); font-weight: 600;">DNN Analyzed</span>
        </div>
        ${cards}
      </div>
    `;
  } else {
    linksHtml = `
      <div class="links-section">
        <div class="links-header"><span>Embedded Hyperlinks</span></div>
        <p style="font-size: 0.85rem; color: var(--text-dim);">No links found in message body. Governed purely by natural language intent.</p>
      </div>
    `;
  }

  container.innerHTML = `
    <!-- Top Verdict Hero -->
    <div class="verdict-banner ${bannerClass}">
      <div class="verdict-icon">
        ${isPhish ? `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
            <line x1="12" y1="9" x2="12" y2="13"></line>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
          </svg>
        ` : `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
          </svg>
        `}
      </div>
      <div class="verdict-text-group">
        <h3>${statusText}</h3>
        <span class="verdict-badge">${data.risk_level} THREAT LEVEL</span>
      </div>
    </div>

    <!-- Gauge & Sub-engine Meters -->
    <div class="metrics-row">
      <div class="gauge-wrapper">
        <svg class="gauge-svg" viewBox="0 0 120 120">
          <circle class="gauge-bg" cx="60" cy="60" r="50"></circle>
          <circle class="gauge-meter" id="gauge-circle" cx="60" cy="60" r="50"></circle>
        </svg>
        <div class="gauge-center-text">
          <div class="gauge-val" id="gauge-number" style="color: ${badgeColor};">0%</div>
          <div class="gauge-label">Risk</div>
        </div>
      </div>

      <div class="sub-scores">
        <div class="score-bar-group">
          <div class="score-bar-header">
            <span>Email NLP Intent (DistilBERT)</span>
            <span class="mono" style="color: var(--accent-cyan);">${emailNLPScore}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill text-nlp" style="width: ${emailNLPScore}%;"></div>
          </div>
        </div>

        ${maxUrlScore !== null ? `
          <div class="score-bar-group">
            <div class="score-bar-header">
              <span>Max Embedded Link Threat (DNN)</span>
              <span class="mono" style="color: var(--accent-rose);">${maxUrlScore}%</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill url-dnn" style="width: ${maxUrlScore}%;"></div>
            </div>
          </div>
        ` : ""}
      </div>
    </div>

    <!-- AI Cybersecurity Reason & Signals -->
    <div class="reason-card">
      <div class="reason-title">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
        <span>Multimodal Decision Rationale</span>
      </div>
      <div class="reason-content">${escapeHtml(data.explanation || "")}</div>
    </div>

    <!-- Links Breakdown -->
    ${linksHtml}
  `;

  // Animate Gauge
  animateGauge(pct, isPhish);
}

// -----------------------------------------------------------------------------
// 4. URL Form Handler
// -----------------------------------------------------------------------------
function initUrlForm() {
  const form = document.getElementById("url-form");
  const submitBtn = document.getElementById("btn-scan-url");
  const clearBtn = document.getElementById("btn-clear-url");
  const resultBox = document.getElementById("url-result-box");

  clearBtn.addEventListener("click", () => {
    document.getElementById("url-input").value = "";
    resetResultBox(resultBox, "Enter any URL on the left to extract 49 structural features and compute its threat score.");
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = document.getElementById("url-input").value.trim();
    if (!url) return;

    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span class="spinner"></span> <span>Inspecting 49 Lexical Features...</span>`;

    resultBox.innerHTML = `
      <div class="empty-state">
        <span class="spinner" style="width: 32px; height: 32px; border-width: 3px; margin-bottom: 14px;"></span>
        <p style="color: var(--accent-cyan); font-weight: 500;">Computing Shannon entropy, token counts, and DNN inference...</p>
      </div>
    `;

    try {
      const res = await fetch("/api/predict/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "URL Analysis failed.");

      renderUrlResult(resultBox, data);
    } catch (err) {
      resultBox.innerHTML = `
        <div class="verdict-banner phish" style="border-color: var(--border-danger);">
          <div class="verdict-text-group">
            <h3>Scan Error</h3>
            <p style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">${err.message}</p>
          </div>
        </div>
      `;
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span class="btn-text">Inspect URL Security</span>`;
    }
  });
}

function renderUrlResult(container, data) {
  const isPhish = data.verdict === "PHISHING";
  const prob = data.combined_phishing_prob;
  const pct = Math.round(prob * 100);

  const urlData = (data.url_analysis && data.url_analysis[0]) || {};
  const signals = urlData.key_signals || [];

  let bannerClass = isPhish ? "phish" : "safe";
  let statusText = isPhish ? "🚨 Malicious Phishing URL" : "🛡️ Legitimate / Safe Link";
  let badgeColor = isPhish ? "var(--accent-rose)" : "var(--accent-emerald)";

  let signalsHtml = "";
  if (signals.length > 0) {
    signalsHtml = `
      <div class="reason-card" style="margin-top: 14px;">
        <div class="reason-title">Flagged Lexical Risk Signals</div>
        <div class="signals-tag-list" style="margin-top: 8px;">
          ${signals.map(s => `<span class="signal-tag" style="background: rgba(255, 51, 102, 0.15); border-color: rgba(255, 51, 102, 0.3); color: #fff;">${s}</span>`).join("")}
        </div>
      </div>
    `;
  }

  container.innerHTML = `
    <div class="verdict-banner ${bannerClass}">
      <div class="verdict-icon">
        ${isPhish ? `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="4.93" y1="4.93" x2="19.07" y2="19.07"></line>
          </svg>
        ` : `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
          </svg>
        `}
      </div>
      <div class="verdict-text-group">
        <h3>${statusText}</h3>
        <span class="verdict-badge">${data.risk_level} THREAT</span>
      </div>
    </div>

    <div class="metrics-row">
      <div class="gauge-wrapper">
        <svg class="gauge-svg" viewBox="0 0 120 120">
          <circle class="gauge-bg" cx="60" cy="60" r="50"></circle>
          <circle class="gauge-meter" id="gauge-circle-url" cx="60" cy="60" r="50"></circle>
        </svg>
        <div class="gauge-center-text">
          <div class="gauge-val" id="gauge-number-url" style="color: ${badgeColor};">0%</div>
          <div class="gauge-label">Threat</div>
        </div>
      </div>

      <div class="sub-scores">
        <div class="score-bar-group">
          <div class="score-bar-header">
            <span>DNN Malicious Probability</span>
            <span class="mono" style="color: ${badgeColor};">${pct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill url-dnn" style="width: ${pct}%;"></div>
          </div>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">
          Scored across 49 features: length, Shannon entropy, TLD risk, IP host verification, and character ratios.
        </div>
      </div>
    </div>

    ${signalsHtml}

    <div class="reason-card" style="margin-top: 14px;">
      <div class="reason-title">Scanned URL</div>
      <div class="mono" style="font-size: 0.85rem; color: var(--accent-cyan); word-break: break-all;">
        ${escapeHtml(urlData.url || "")}
      </div>
    </div>
  `;

  animateGauge(pct, isPhish, "gauge-circle-url", "gauge-number-url");
}

// -----------------------------------------------------------------------------
// 5. Gauge Animation Helper
// -----------------------------------------------------------------------------
function animateGauge(targetPct, isPhish, circleId = "gauge-circle", numId = "gauge-number") {
  const circle = document.getElementById(circleId);
  const numElem = document.getElementById(numId);
  if (!circle || !numElem) return;

  const circumference = 314; // 2 * pi * 50
  const offset = circumference - (targetPct / 100) * circumference;

  circle.style.stroke = isPhish ? "var(--accent-rose)" : "var(--accent-emerald)";
  circle.style.strokeDashoffset = offset;

  // Number count-up
  let current = 0;
  const duration = 700;
  const stepTime = 16;
  const steps = duration / stepTime;
  const increment = targetPct / steps;

  const timer = setInterval(() => {
    current += increment;
    if (current >= targetPct) {
      current = targetPct;
      clearInterval(timer);
    }
    numElem.textContent = Math.round(current) + "%";
  }, stepTime);
}

// -----------------------------------------------------------------------------
// 6. Benchmark Matrix Loader
// -----------------------------------------------------------------------------
async function loadBenchmarks() {
  const tableBody = document.getElementById("benchmark-table-body");
  if (!tableBody) return;

  try {
    const res = await fetch("/api/benchmarks");
    if (!res.ok) return;
    const data = await res.json();

    if (data.models && data.models.length > 0) {
      tableBody.innerHTML = data.models.map((m, idx) => {
        const isHighlight = idx === data.models.length - 1;
        const rowClass = isHighlight ? "highlight-row" : "";
        return `
          <tr class="${rowClass}">
            <td><strong>${m.name}</strong><br><span style="font-size: 0.78rem; color: var(--text-dim);">${m.framework}</span></td>
            <td>${m.scope}</td>
            <td class="mono">${m.samples.toLocaleString()}</td>
            <td class="score-badge-green mono">${m.accuracy.toFixed(2)}%</td>
            <td class="mono">${m.precision.toFixed(2)}%</td>
            <td class="mono">${m.recall.toFixed(2)}%</td>
            <td class="score-badge-blue mono">${m.f1.toFixed(4)}</td>
            <td class="score-badge-blue mono">${m.roc_auc.toFixed(4)}</td>
          </tr>
        `;
      }).join("");
    }
  } catch (err) {
    console.warn("Could not load benchmarks:", err);
  }
}

// -----------------------------------------------------------------------------
// 7. Utility Helpers
// -----------------------------------------------------------------------------
function resetResultBox(box, emptyMessage) {
  box.innerHTML = `
    <div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="1.5">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="8" x2="12" y2="12"></line>
        <line x1="12" y1="16" x2="12.01" y2="16"></line>
      </svg>
      <p>${emptyMessage}</p>
    </div>
  `;
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
