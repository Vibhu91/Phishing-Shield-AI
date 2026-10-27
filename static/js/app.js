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
  initAttachmentScanner();
  initHeadersScanner();
  initEnterpriseGateway();
  loadFeedbackTriage();
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

    <!-- Analyst Feedback & Triage Action Bar -->
    <div style="margin-top: 20px; padding: 14px; background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: 8px;">
      <div style="font-size: 0.8rem; font-weight: 600; color: var(--text-dim); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">
        SOC Analyst Feedback (Human-In-The-Loop Triage)
      </div>
      <div style="display: flex; gap: 8px; flex-wrap: wrap;" id="analyst-feedback-btns">
        <button type="button" class="btn btn-clear" style="font-size: 0.8rem; padding: 6px 12px; border-color: rgba(244, 63, 94, 0.4); color: var(--accent-rose);" onclick="submitAnalystFeedback('email', ${prob}, '${data.verdict}', 'PHISHING')">
          🚨 Confirm Phishing Threat
        </button>
        <button type="button" class="btn btn-clear" style="font-size: 0.8rem; padding: 6px 12px; border-color: rgba(56, 189, 248, 0.4); color: var(--accent-cyan);" onclick="submitAnalystFeedback('email', ${prob}, '${data.verdict}', 'LEGITIMATE')">
          ⚠️ Report False Positive (Legitimate)
        </button>
      </div>
      <div id="analyst-feedback-feedback-msg" style="margin-top: 6px; font-size: 0.8rem; display: none;"></div>
    </div>
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

// -----------------------------------------------------------------------------
// 8. Attachment & Quishing Scanner Controller
// -----------------------------------------------------------------------------
function initAttachmentScanner() {
  const form = document.getElementById("attachment-form");
  const fileInput = document.getElementById("attachment-file-input");
  const resultBox = document.getElementById("attachment-result-box");
  const btnQrDemo = document.getElementById("btn-load-qr-demo");
  const btnPdfDemo = document.getElementById("btn-load-pdf-demo");
  const clearBtn = document.getElementById("btn-clear-attachment");

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      fileInput.value = "";
      resetResultBox(resultBox, "Upload an attachment or select a preset to begin Quishing inspection.");
    });
  }

  if (btnQrDemo) {
    btnQrDemo.addEventListener("click", async () => {
      resultBox.innerHTML = "<p style='color: var(--accent-cyan);'>Decoding synthetic Quishing flyer...</p>";
      try {
        // Request backend to test synthetic QR code or send sample
        const res = await fetch("/api/scan/attachment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filename: "quishing_alert_sample.png",
            file_b64: "" // server fallback handles or tests
          })
        });
        const data = await res.json();
        renderAttachmentResult(resultBox, data);
      } catch (err) {
        resultBox.innerHTML = `<p style="color: var(--accent-rose);">Scan failed: ${escapeHtml(err.message)}</p>`;
      }
    });
  }

  if (btnPdfDemo) {
    btnPdfDemo.addEventListener("click", async () => {
      resultBox.innerHTML = "<p style='color: var(--accent-cyan);'>Inspecting PDF URI actions and JavaScript streams...</p>";
      try {
        const samplePdfBytes = "JVBERi0xLjQKMSAwIG9iajw8L1R5cGUvQ2F0YWxvZy9QYWdlcyAyIDAgUj4+ZW5kb2JqCjIgMCBvYmo8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PmVuZG9iagozIDAgb2JqPDwvVHlwZS9QYWdlL1BhcmVudCAyIDAgUi9Bbm5vdHNbNCAwIFJdPj5lbmRvYmoKNCAwIG9iajw8L1R5cGUvQW5ub3QvU3VidHlwZS9MaW5rL0E8PC9TL1VSSS9VUkkoaHR0cDovL21hbGljaW91cy1sb2dpbi1wb3J0YWwueHl6L2F1dGgpPj4+PmVuZG9iago1IDAgb2JqPDwvSmF2YVNjcmlwdChhcHAuYWxlcnQoJ01hbHdhcmUnKTspPj5lbmRvYmoKdHJhaWxlcjw8L1Jvb3QgMSAwIFI+PgolJUVPRg==";
        const res = await fetch("/api/scan/attachment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filename: "invoice_update.pdf",
            file_b64: samplePdfBytes
          })
        });
        const data = await res.json();
        renderAttachmentResult(resultBox, data);
      } catch (err) {
        resultBox.innerHTML = `<p style="color: var(--accent-rose);">Scan failed: ${escapeHtml(err.message)}</p>`;
      }
    });
  }

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!fileInput.files || fileInput.files.length === 0) {
        alert("Please choose a file to inspect.");
        return;
      }
      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append("file", file);

      resultBox.innerHTML = "<p style='color: var(--accent-cyan);'>Analyzing attachment for QR codes, links &amp; payloads...</p>";
      try {
        const res = await fetch("/api/scan/attachment", {
          method: "POST",
          body: formData
        });
        const data = await res.json();
        renderAttachmentResult(resultBox, data);
      } catch (err) {
        resultBox.innerHTML = `<p style="color: var(--accent-rose);">Upload scan failed: ${escapeHtml(err.message)}</p>`;
      }
    });
  }
}

function renderAttachmentResult(container, data) {
  const hasThreat = data.has_threat || (data.url_assessments && data.url_assessments.some(u => u.verdict === "PHISHING"));
  const bannerClass = hasThreat ? "phish" : "safe";
  const title = hasThreat ? "🚨 Attachment Threat / Quishing Detected" : "🛡️ Clean Attachment";

  let flagsHtml = "";
  if (data.threat_flags && data.threat_flags.length > 0) {
    flagsHtml = data.threat_flags.map(f => `<li style="margin-bottom: 6px; color: var(--accent-rose);">${escapeHtml(f)}</li>`).join("");
  } else {
    flagsHtml = "<li style='color: var(--accent-emerald);'>No malicious structures or QR payloads discovered.</li>";
  }

  let urlsHtml = "";
  if (data.url_assessments && data.url_assessments.length > 0) {
    urlsHtml = data.url_assessments.map(u => `
      <div class="link-card ${u.verdict === 'PHISHING' ? 'link-danger' : ''}" style="margin-top: 8px;">
        <div class="link-top-row">
          <span class="mono">${escapeHtml(u.url)}</span>
          <span class="link-risk-pill ${u.verdict === 'PHISHING' ? 'danger' : 'safe'}">${Math.round(u.phishing_prob * 100)}% Risk</span>
        </div>
      </div>
    `).join("");
  }

  container.innerHTML = `
    <div class="verdict-banner ${bannerClass}">
      <div class="verdict-text-block">
        <h3 class="verdict-title">${title}</h3>
        <p class="verdict-subtext">File: ${escapeHtml(data.filename)} (${escapeHtml(data.extension)})</p>
      </div>
    </div>
    <div class="reason-card" style="margin-top: 16px;">
      <div class="reason-title">Structural &amp; QR Indicators</div>
      <ul style="padding-left: 20px; font-size: 0.88rem; margin-top: 8px;">${flagsHtml}</ul>
    </div>
    ${urlsHtml ? `<div style="margin-top: 14px;"><div style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted); margin-bottom: 6px;">Discovered Web Payloads (URL DNN Evaluated):</div>${urlsHtml}</div>` : ""}
  `;
}

// -----------------------------------------------------------------------------
// 9. Email Header Security Scanner Controller
// -----------------------------------------------------------------------------
function initHeadersScanner() {
  const form = document.getElementById("headers-form");
  const input = document.getElementById("headers-input");
  const resultBox = document.getElementById("headers-result-box");
  const btnSpoof = document.getElementById("btn-load-header-spoof");
  const btnLegit = document.getElementById("btn-load-header-legit");
  const clearBtn = document.getElementById("btn-clear-headers");

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      input.value = "";
      resetResultBox(resultBox, "Paste email headers and click Audit to inspect authentication records.");
    });
  }

  if (btnSpoof) {
    btnSpoof.addEventListener("click", () => {
      input.value = `From: "PayPal Account Security" <service-alert@paypal-update-alert.xyz>
Return-Path: <bounce@unregistered-relay.org>
Reply-To: <credentials-harvest@gmail.com>
Authentication-Results: mx.google.com;
    spf=fail (google.com: domain of service-alert@paypal-update-alert.xyz does not designate 192.0.2.1 as permitted sender);
    dkim=fail header.i=@paypal-update-alert.xyz;
    dmarc=fail (p=REJECT sp=REJECT dis=NONE) header.from=paypal.com
Subject: URGENT: Your PayPal Account Has Been Suspended`;
    });
  }

  if (btnLegit) {
    btnLegit.addEventListener("click", () => {
      input.value = `From: "Microsoft 365 Security" <no-reply@microsoft.com>
Return-Path: <ms-bounces@microsoft.com>
Authentication-Results: spf=pass (sender IP is 40.107.1.1) smtp.mailfrom=microsoft.com;
    dkim=pass header.d=microsoft.com;
    dmarc=pass (p=REJECT) header.from=microsoft.com
Subject: Security notice: Sign-in from new country`;
    });
  }

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const raw = input.value.trim();
      if (!raw) {
        alert("Please paste email headers to audit.");
        return;
      }
      resultBox.innerHTML = "<p style='color: var(--accent-cyan);'>Validating SPF, DKIM, DMARC, and sender authenticity...</p>";
      try {
        const res = await fetch("/api/scan/headers", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ headers: raw })
        });
        const data = await res.json();
        renderHeadersResult(resultBox, data);
      } catch (err) {
        resultBox.innerHTML = `<p style="color: var(--accent-rose);">Header audit failed: ${escapeHtml(err.message)}</p>`;
      }
    });
  }
}

function renderHeadersResult(container, data) {
  const isSuspicious = data.header_risk_score >= 0.40;
  const bannerClass = isSuspicious ? "phish" : "safe";
  const title = isSuspicious ? `🚨 ${data.header_verdict}: Spoofing / Authentication Failure` : `🛡️ ${data.header_verdict}: Headers Validated`;

  const spfColor = data.spf.status === "pass" ? "var(--accent-emerald)" : (data.spf.status === "fail" ? "var(--accent-rose)" : "var(--text-muted)");
  const dkimColor = (data.dkim.status === "pass" || data.dkim.status === "present") ? "var(--accent-emerald)" : (data.dkim.status === "fail" ? "var(--accent-rose)" : "var(--text-muted)");
  const dmarcColor = data.dmarc.status === "pass" ? "var(--accent-emerald)" : (data.dmarc.status === "fail" ? "var(--accent-rose)" : "var(--text-muted)");

  let anomaliesHtml = "";
  if (data.anomalies && data.anomalies.length > 0) {
    anomaliesHtml = data.anomalies.map(a => `<li style="margin-bottom: 6px; color: var(--accent-rose);">${escapeHtml(a)}</li>`).join("");
  } else {
    anomaliesHtml = "<li style='color: var(--accent-emerald);'>All cryptographic authentication records passed alignment.</li>";
  }

  container.innerHTML = `
    <div class="verdict-banner ${bannerClass}">
      <div class="verdict-text-block">
        <h3 class="verdict-title">${title}</h3>
        <p class="verdict-subtext">Sender: ${escapeHtml(data.from || "Unknown")} (Domain: ${escapeHtml(data.from_domain)})</p>
      </div>
    </div>

    <div style="display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap;">
      <div style="flex: 1; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid var(--border-color); text-align: center;">
        <span style="font-size: 0.75rem; color: var(--text-muted);">SPF Status</span>
        <div style="font-weight: 700; color: ${spfColor}; text-transform: uppercase; margin-top: 2px;">${escapeHtml(data.spf.status)}</div>
      </div>
      <div style="flex: 1; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid var(--border-color); text-align: center;">
        <span style="font-size: 0.75rem; color: var(--text-muted);">DKIM Status</span>
        <div style="font-weight: 700; color: ${dkimColor}; text-transform: uppercase; margin-top: 2px;">${escapeHtml(data.dkim.status)}</div>
      </div>
      <div style="flex: 1; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid var(--border-color); text-align: center;">
        <span style="font-size: 0.75rem; color: var(--text-muted);">DMARC Status</span>
        <div style="font-weight: 700; color: ${dmarcColor}; text-transform: uppercase; margin-top: 2px;">${escapeHtml(data.dmarc.status)}</div>
      </div>
    </div>

    ${data.spoofed_brand ? `
      <div style="margin-top: 14px; padding: 12px; background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.4); border-radius: 6px; color: #fecdd3; font-size: 0.85rem;">
        <strong>⚠️ Display Name Impersonation:</strong> This message impersonates <strong>${escapeHtml(data.spoofed_brand)}</strong>, but originates from domain <code>${escapeHtml(data.from_domain)}</code>.
      </div>
    ` : ""}

    <div class="reason-card" style="margin-top: 14px;">
      <div class="reason-title">Security &amp; Alignment Findings</div>
      <ul style="padding-left: 20px; font-size: 0.88rem; margin-top: 8px;">${anomaliesHtml}</ul>
    </div>
  `;
}

// -----------------------------------------------------------------------------
// 10. Enterprise Mailbox Gateway & Continuous Retraining Controller
// -----------------------------------------------------------------------------
function initEnterpriseGateway() {
  const btnScan = document.getElementById("btn-trigger-gateway-scan");
  const output = document.getElementById("gateway-scan-output");
  const btnRetrain = document.getElementById("btn-trigger-retraining");
  const btnRefresh = document.getElementById("btn-refresh-feedback");

  if (btnScan) {
    btnScan.addEventListener("click", async () => {
      output.innerHTML = "<p style='color: var(--accent-cyan);'>Polling Microsoft 365 Graph API &amp; Gmail Mailboxes...</p>";
      try {
        const res = await fetch("/api/gateway/scan");
        const data = await res.json();
        let logHtml = "";
        if (data.threat_log && data.threat_log.length > 0) {
          logHtml = data.threat_log.map(item => `
            <div style="padding: 10px; margin-bottom: 8px; background: rgba(244, 63, 94, 0.08); border-left: 3px solid var(--accent-rose); border-radius: 4px;">
              <strong>[${escapeHtml(item.gateway)}]</strong> Action: <span style="color: var(--accent-rose); font-weight: 700;">${escapeHtml(item.action)}</span><br>
              <span style="font-size: 0.85rem;">${escapeHtml(item.subject)}</span> (Sender: <code>${escapeHtml(item.sender)}</code>) — Threat: ${(item.phishing_prob*100).toFixed(1)}%
            </div>
          `).join("");
        } else {
          logHtml = "<p style='color: var(--accent-emerald); font-size: 0.9rem;'>All polled inbox messages verified safe. Zero threats detected.</p>";
        }

        output.innerHTML = `
          <div style="margin-top: 14px; padding: 14px; background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: 8px;">
            <div style="font-weight: 600; margin-bottom: 8px;">Gateway Poll Results: M365 Scanned: ${data.m365_scanned} (${data.m365_quarantined} Quarantined) | Gmail Scanned: ${data.gmail_scanned} (${data.gmail_flagged} Flagged)</div>
            ${logHtml}
          </div>
        `;
      } catch (err) {
        output.innerHTML = `<p style="color: var(--accent-rose);">Gateway sync error: ${escapeHtml(err.message)}</p>`;
      }
    });
  }

  if (btnRetrain) {
    btnRetrain.addEventListener("click", async () => {
      const alertBox = document.getElementById("retrain-status-alert");
      alertBox.innerHTML = "<p style='color: var(--accent-cyan);'>Triggering incremental training update...</p>";
      try {
        const res = await fetch("/api/retrain/trigger", { method: "POST" });
        const data = await res.json();
        alertBox.innerHTML = `<div style="padding: 10px; background: rgba(16, 185, 129, 0.1); border: 1px solid var(--accent-emerald); border-radius: 6px; color: #a7f3d0; margin-bottom: 12px;">✅ ${escapeHtml(data.message)}</div>`;
        loadFeedbackTriage();
      } catch (err) {
        alertBox.innerHTML = `<p style="color: var(--accent-rose);">Retraining failed: ${escapeHtml(err.message)}</p>`;
      }
    });
  }

  if (btnRefresh) {
    btnRefresh.addEventListener("click", () => {
      loadFeedbackTriage();
    });
  }
}

async function loadFeedbackTriage() {
  const statsContainer = document.getElementById("feedback-stats-counters");
  const tableBody = document.querySelector("#feedback-triage-table tbody");
  if (!statsContainer || !tableBody) return;

  try {
    const res = await fetch("/api/feedback/stats");
    if (!res.ok) return;
    const data = await res.json();

    statsContainer.innerHTML = `
      <div style="flex: 1; min-width: 130px; padding: 12px; background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: 6px;">
        <div style="font-size: 0.75rem; color: var(--text-muted);">Total Triage Feedback</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #fff;">${data.total_feedback}</div>
      </div>
      <div style="flex: 1; min-width: 130px; padding: 12px; background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: 6px;">
        <div style="font-size: 0.75rem; color: var(--text-muted);">False Positives Logged</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: var(--accent-cyan);">${data.false_positives}</div>
      </div>
      <div style="flex: 1; min-width: 130px; padding: 12px; background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: 6px;">
        <div style="font-size: 0.75rem; color: var(--text-muted);">False Negatives Logged</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: var(--accent-rose);">${data.false_negatives}</div>
      </div>
      <div style="flex: 1; min-width: 130px; padding: 12px; background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: 6px;">
        <div style="font-size: 0.75rem; color: var(--text-muted);">Pending Retrain Queue</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: var(--accent-amber);">${data.pending_retrain_samples}</div>
      </div>
    `;

    if (data.recent_feedback && data.recent_feedback.length > 0) {
      tableBody.innerHTML = data.recent_feedback.map(row => `
        <tr>
          <td class="mono">#${row.id}</td>
          <td><span class="signal-tag">${escapeHtml(row.type)}</span></td>
          <td style="max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(row.content)}</td>
          <td><span style="color: ${row.predicted === 'PHISHING' ? 'var(--accent-rose)' : 'var(--accent-emerald)'}; font-weight: 600;">${row.predicted}</span></td>
          <td><span style="color: ${row.analyst === 'PHISHING' ? 'var(--accent-rose)' : 'var(--accent-emerald)'}; font-weight: 600;">${row.analyst}</span></td>
          <td><span style="font-size: 0.78rem; font-weight: 700; color: ${row.feedback_type === 'FALSE_POSITIVE' ? 'var(--accent-cyan)' : (row.feedback_type === 'FALSE_NEGATIVE' ? 'var(--accent-rose)' : 'var(--accent-emerald)')};">${escapeHtml(row.feedback_type)}</span></td>
          <td>${row.is_retrained ? '<span style="color: var(--accent-emerald); font-size: 0.8rem;">Retrained</span>' : '<span style="color: var(--accent-amber); font-size: 0.8rem;">Pending</span>'}</td>
        </tr>
      `).join("");
    } else {
      tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 16px;">No analyst triage feedback recorded yet.</td></tr>`;
    }
  } catch (err) {
    console.warn("Could not load feedback triage:", err);
  }
}

async function submitAnalystFeedback(sampleType, predProb, predVerdict, analystVerdict) {
  const content = sampleType === "email" ? (document.getElementById("email-body").value || document.getElementById("email-subject").value) : (document.getElementById("url-input").value);
  const msgBox = document.getElementById("analyst-feedback-feedback-msg");

  try {
    const res = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_type: sampleType,
        target_content: content,
        predicted_prob: predProb,
        predicted_verdict: predVerdict,
        analyst_verdict: analystVerdict,
        analyst_notes: `SOC Analyst verified ${analystVerdict}`
      })
    });
    const data = await res.json();
    if (msgBox) {
      msgBox.style.display = "block";
      msgBox.style.color = "var(--accent-emerald)";
      msgBox.innerHTML = `✅ ${escapeHtml(data.message)} Added to continuous retraining queue.`;
    }
    loadFeedbackTriage();
  } catch (err) {
    alert("Could not log feedback: " + err.message);
  }
}

