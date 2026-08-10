(() => {
  "use strict";

  // Localhost → dev FastAPI on port 2200
  // Vercel (or any other host) → same-origin, FastAPI serves everything
  const API_BASE =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
      ? "http://127.0.0.1:2200"
      : "";

  const form          = document.getElementById("predict-form");
  const submitBtn     = document.getElementById("submit-btn");
  const resetBtn      = document.getElementById("reset-btn");
  const errorRetryBtn = document.getElementById("error-retry-btn");

  const stateIdle    = document.getElementById("state-idle");
  const stateLoading = document.getElementById("state-loading");
  const stateResult  = document.getElementById("state-result");
  const stateError   = document.getElementById("state-error");

  const scoreNumberEl = document.getElementById("score-number");
  const scoreBandEl   = document.getElementById("score-band");
  const scoreContextEl= document.getElementById("score-context");
  const gaugeFill     = document.getElementById("gauge-fill");
  const errorCopyEl   = document.getElementById("error-copy");

  // Recommendations panel elements
  const riskSection    = document.getElementById("risk-section");
  const riskList       = document.getElementById("risk-list");
  const posSection     = document.getElementById("positives-section");
  const posList        = document.getElementById("positives-list");
  const recsSection    = document.getElementById("recs-section");
  const recsGrid       = document.getElementById("recs-grid");

  const GAUGE_ARC_LENGTH = 314; // approx pi * r(100)

  // ---------------------------------------------------------
  // Draw tick marks on both gauges (0..10, every 2 units)
  // ---------------------------------------------------------
  function drawTicks() {
    document.querySelectorAll(".gauge-ticks").forEach((g) => {
      g.innerHTML = "";
      const cx = 120, cy = 140, rOuter = 100, rInner = 90;
      for (let i = 0; i <= 10; i += 2) {
        const angle = Math.PI - (i / 10) * Math.PI;
        const x1 = cx + rOuter * Math.cos(angle);
        const y1 = cy - rOuter * Math.sin(angle);
        const x2 = cx + rInner * Math.cos(angle);
        const y2 = cy - rInner * Math.sin(angle);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x1.toFixed(1));
        line.setAttribute("y1", y1.toFixed(1));
        line.setAttribute("x2", x2.toFixed(1));
        line.setAttribute("y2", y2.toFixed(1));
        g.appendChild(line);
      }
    });
  }
  drawTicks();

  // ---------------------------------------------------------
  // Segmented control (stress_level) wiring
  // ---------------------------------------------------------
  const segGroup         = document.getElementById("stress_level_group");
  const stressHiddenInput= document.getElementById("stress_level");
  segGroup.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      segGroup.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      stressHiddenInput.value = btn.dataset.value;
      clearFieldError(stressHiddenInput);
    });
  });

  // ---------------------------------------------------------
  // Field-level error helpers
  // ---------------------------------------------------------
  function fieldWrapper(input) { return input.closest(".field"); }

  function setFieldError(input, message) {
    const wrap = fieldWrapper(input);
    if (!wrap) return;
    wrap.classList.add("field-error");
    const msgEl = wrap.querySelector(".error-msg");
    if (msgEl) msgEl.textContent = message;
  }

  function clearFieldError(input) {
    const wrap = fieldWrapper(input);
    if (!wrap) return;
    wrap.classList.remove("field-error");
    const msgEl = wrap.querySelector(".error-msg");
    if (msgEl) msgEl.textContent = "";
  }

  function clearAllErrors() {
    form.querySelectorAll(".field").forEach((f) => f.classList.remove("field-error"));
    form.querySelectorAll(".error-msg").forEach((m) => (m.textContent = ""));
  }

  // ---------------------------------------------------------
  // Client-side validation mirroring the StudentData model
  // ---------------------------------------------------------
  function validate(payload) {
    const errors = [];
    const numericChecks = [
      ["age",                    10,       100],
      ["avg_daily_usage_hours",   0,        24],
      ["daily_unlocks",           0, Infinity],
      ["study_hours",             0,        24],
      ["physical_activity_hours", 0,        24],
      ["sleep_hours_per_night",   0,        24],
    ];
    numericChecks.forEach(([key, min, max]) => {
      const input = document.getElementById(key);
      const val   = payload[key];
      if (val === "" || val === null || Number.isNaN(val)) {
        errors.push([input, "This field is required."]);
      } else if (val < min || val > max) {
        errors.push([input, `Must be between ${min} and ${max === Infinity ? "0+" : max}.`]);
      }
    });
    ["gender", "country", "academic_level", "most_used_platform", "purpose_of_use"].forEach((key) => {
      const input = document.getElementById(key);
      if (!payload[key] || String(payload[key]).trim() === "") {
        errors.push([input, "This field is required."]);
      }
    });
    if (!payload.stress_level) {
      errors.push([stressHiddenInput, "Pick a stress level."]);
    }
    return errors;
  }

  // ---------------------------------------------------------
  // Gather form data into the exact StudentData shape
  // ---------------------------------------------------------
  function collectPayload() {
    const fd = new FormData(form);
    return {
      age                    : fd.get("age") === "" ? NaN : parseInt(fd.get("age"), 10),
      gender                 : fd.get("gender") || "",
      country                : (fd.get("country") || "").trim(),
      academic_level         : fd.get("academic_level") || "",
      most_used_platform     : fd.get("most_used_platform") || "",
      purpose_of_use         : fd.get("purpose_of_use") || "",
      avg_daily_usage_hours  : fd.get("avg_daily_usage_hours") === "" ? NaN : parseFloat(fd.get("avg_daily_usage_hours")),
      daily_unlocks          : fd.get("daily_unlocks") === "" ? NaN : parseInt(fd.get("daily_unlocks"), 10),
      study_hours            : fd.get("study_hours") === "" ? NaN : parseFloat(fd.get("study_hours")),
      physical_activity_hours: fd.get("physical_activity_hours") === "" ? NaN : parseFloat(fd.get("physical_activity_hours")),
      sleep_hours_per_night  : fd.get("sleep_hours_per_night") === "" ? NaN : parseFloat(fd.get("sleep_hours_per_night")),
      stress_level           : fd.get("stress_level") || "",
    };
  }

  // ---------------------------------------------------------
  // UI state switching
  // ---------------------------------------------------------
  function showState(name) {
    [stateIdle, stateLoading, stateResult, stateError].forEach((el) => (el.hidden = true));
    ({ idle: stateIdle, loading: stateLoading, result: stateResult, error: stateError }[name]).hidden = false;
  }

  function setSubmitting(isSubmitting) {
    submitBtn.disabled = isSubmitting;
    submitBtn.classList.toggle("loading", isSubmitting);
  }

  function bandFor(score, category) {
    if (category === "strained") {
      return {
        label: "Signal: strained",
        context: "Your responses suggest elevated strain right now. Small, consistent shifts in sleep, activity, and screen time can make a real difference.",
      };
    }
    if (category === "balanced") {
      return {
        label: "Signal: balanced",
        context: "Your rhythm looks fairly steady, with some room to recover and reset. Keep building on your strengths.",
      };
    }
    return {
      label: "Signal: strong",
      context: "Your habits point to a well-supported, resilient baseline. Keep it up and protect what's working.",
    };
  }

  // ---------------------------------------------------------
  // Staggered animation helper
  // ---------------------------------------------------------
  function animateIn(elements, delayStep = 80) {
    elements.forEach((el, i) => {
      el.style.opacity    = "0";
      el.style.transform  = "translateY(12px)";
      el.style.transition = "none";
      setTimeout(() => {
        el.style.transition = "opacity 0.35s ease, transform 0.35s ease";
        el.style.opacity    = "1";
        el.style.transform  = "translateY(0)";
      }, i * delayStep + 30);
    });
  }

  // ---------------------------------------------------------
  // Render the full analysis result
  // ---------------------------------------------------------
  function renderResult(data) {
    const { score, category, risk_factors, positives, recommendations } = data;
    const clamped = Math.max(0, Math.min(10, score));
    const { label, context } = bandFor(clamped, category);

    // Score display
    scoreNumberEl.textContent = score.toFixed(2);
    scoreBandEl.textContent   = label;
    scoreBandEl.className     = `score-band score-band--${category}`;
    scoreContextEl.textContent= context;

    // Gauge animation
    gaugeFill.style.transition = "none";
    gaugeFill.style.strokeDashoffset = String(GAUGE_ARC_LENGTH);
    requestAnimationFrame(() => {
      gaugeFill.style.transition = "";
      const offset = GAUGE_ARC_LENGTH * (1 - clamped / 10);
      gaugeFill.style.strokeDashoffset = String(offset);
    });

    // ── Risk factors ──────────────────────────────────────────────────────
    if (risk_factors && risk_factors.length > 0) {
      riskList.innerHTML = "";
      risk_factors.forEach((rf) => {
        const li = document.createElement("li");
        li.className = "risk-item";
        li.innerHTML = `<span class="risk-dot"></span><span>${rf}</span>`;
        riskList.appendChild(li);
      });
      riskSection.hidden = false;
      animateIn([...riskList.querySelectorAll(".risk-item")]);
    } else {
      riskSection.hidden = true;
    }

    // ── Positives ─────────────────────────────────────────────────────────
    if (positives && positives.length > 0) {
      posList.innerHTML = "";
      positives.forEach((p) => {
        const li = document.createElement("li");
        li.className = "positive-item";
        li.innerHTML = `<span class="positive-dot"></span><span>${p}</span>`;
        posList.appendChild(li);
      });
      posSection.hidden = false;
      animateIn([...posList.querySelectorAll(".positive-item")], 60);
    } else {
      posSection.hidden = true;
    }

    // ── Recommendations ───────────────────────────────────────────────────
    if (recommendations && recommendations.length > 0) {
      recsGrid.innerHTML = "";
      recommendations.forEach((rec) => {
        const card = document.createElement("div");
        card.className = "rec-card";
        card.innerHTML = `
          <div class="rec-icon">${rec.icon}</div>
          <div class="rec-body">
            <p class="rec-title">${rec.title}</p>
            <p class="rec-detail">${rec.detail}</p>
          </div>`;
        recsGrid.appendChild(card);
      });
      recsSection.hidden = false;
      animateIn([...recsGrid.querySelectorAll(".rec-card")], 100);
    } else {
      recsSection.hidden = true;
    }

    showState("result");
  }

  function renderError(copy) {
    errorCopyEl.textContent = copy;
    showState("error");
  }

  // ---------------------------------------------------------
  // Parse FastAPI / Pydantic 422 error responses
  // ---------------------------------------------------------
  function applyServerValidationErrors(detail) {
    if (!Array.isArray(detail)) return false;
    let matched = false;
    detail.forEach((err) => {
      const field  = Array.isArray(err.loc) ? err.loc[err.loc.length - 1] : null;
      const input  = field ? document.getElementById(field) : null;
      const target = field === "stress_level" ? stressHiddenInput : input;
      if (target) {
        setFieldError(target, err.msg || "Invalid value.");
        matched = true;
      }
    });
    return matched;
  }

  // ---------------------------------------------------------
  // Submit handler
  // ---------------------------------------------------------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearAllErrors();

    const payload      = collectPayload();
    const clientErrors = validate(payload);

    if (clientErrors.length > 0) {
      clientErrors.forEach(([input, msg]) => input && setFieldError(input, msg));
      clientErrors[0][0]?.focus?.();
      return;
    }

    setSubmitting(true);
    showState("loading");

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify(payload),
      });

      if (res.status === 422) {
        const body    = await res.json().catch(() => null);
        const matched = body && applyServerValidationErrors(body.detail);
        renderError(
          matched
            ? "The API rejected a few fields — details are marked on the form."
            : "The API rejected this submission. Please review your inputs and try again."
        );
        return;
      }

      if (!res.ok) {
        let detailMsg = `The API responded with status ${res.status}.`;
        const body = await res.json().catch(() => null);
        if (body && typeof body.detail === "string") detailMsg = body.detail;
        renderError(detailMsg);
        return;
      }

      const data = await res.json();
      if (typeof data.score !== "number") {
        renderError("The API responded, but the score was missing or malformed.");
        return;
      }

      renderResult(data);
    } catch (err) {
      renderError(
        `Couldn't connect to ${API_BASE}. Make sure the backend is running (uvicorn main:app --port 2200 --reload) and reachable from this page.`
      );
    } finally {
      setSubmitting(false);
    }
  });

  // live-clear errors as the user edits
  form.querySelectorAll("input, select").forEach((el) => {
    el.addEventListener("input",  () => clearFieldError(el));
    el.addEventListener("change", () => clearFieldError(el));
  });

  resetBtn.addEventListener("click", () => {
    // Clear recommendation panels before going back to idle
    riskSection.hidden  = true;
    posSection.hidden   = true;
    recsSection.hidden  = true;
    showState("idle");
  });

  errorRetryBtn.addEventListener("click", () => { showState("idle"); });
})();
