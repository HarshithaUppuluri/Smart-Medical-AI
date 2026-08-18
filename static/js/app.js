// MediAssist AI frontend — framework-free interactions for the Flask JSON API.
(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  // Theme -------------------------------------------------------------------
  const THEME_KEY = "carelens-theme";
  const themeToggle = $("#theme-toggle");
  const storedTheme = localStorage.getItem(THEME_KEY);
  if (storedTheme) document.documentElement.dataset.theme = storedTheme;

  themeToggle.addEventListener("click", () => {
    const explicit = document.documentElement.dataset.theme;
    const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const currentDark = explicit ? explicit === "dark" : systemDark;
    const next = currentDark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem(THEME_KEY, next);
    themeToggle.setAttribute("aria-label", `Switch to ${currentDark ? "dark" : "light"} theme`);
  });

  // Tabs --------------------------------------------------------------------
  const tabs = $$(".tab");
  const panels = $$(".panel");

  function activateTab(name, moveFocus = false) {
    tabs.forEach((tab) => {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      if (active && moveFocus) tab.focus();
    });
    panels.forEach((panel) => {
      const active = panel.id === `panel-${name}`;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
    });
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let next = index;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      activateTab(tabs[next].dataset.tab, true);
    });
  });
  activateTab("symptoms");

  // Shared helpers ----------------------------------------------------------
  const templates = {
    confident: $("#tpl-confident"),
    unconfident: $("#tpl-unconfident"),
    bar: $("#tpl-bar"),
    error: $("#tpl-error"),
  };
  const toast = $("#toast");
  let toastTimer;

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 2200);
  }

  function buildBars(container, entries, topLabel) {
    container.innerHTML = "";
    entries.forEach((entry) => {
      const node = templates.bar.content.cloneNode(true);
      const row = $(".bar-row", node);
      const pct = entry.prob * 100;
      $('[data-slot="label"]', row).textContent = entry.label;
      $('[data-slot="value"]', row).textContent = `${pct.toFixed(1)}%`;
      const fill = $('[data-slot="fill"]', row);
      if (entry.label === topLabel) row.classList.add("is-top");
      container.appendChild(node);
      requestAnimationFrame(() => { fill.style.width = `${pct}%`; });
    });
  }

  function renderConfident(container, data, { showInfo = false } = {}) {
    const node = templates.confident.content.cloneNode(true);
    $('[data-slot="prediction"]', node).textContent = data.prediction;
    $('[data-slot="confidence"]', node).textContent = (data.confidence * 100).toFixed(1);
    buildBars($('[data-slot="bars"]', node), data.top3, data.prediction);
    if (showInfo && data.info && (data.info.description || data.info.precautions?.length)) {
      const info = $('[data-slot="info"]', node);
      info.hidden = false;
      if (data.info.description) $('[data-slot="description"]', node).textContent = data.info.description;
      if (data.info.precautions?.length) {
        const wrap = $('[data-slot="precautions"]', node);
        const title = document.createElement("p");
        title.className = "precaution-title";
        title.textContent = "Suggested precautions";
        const list = document.createElement("ul");
        list.className = "precaution-list";
        data.info.precautions.forEach((item) => {
          const li = document.createElement("li");
          li.textContent = item;
          list.appendChild(li);
        });
        wrap.append(title, list);
      }
    }
    container.replaceChildren(node);
    return $(".result", container);
  }

  function renderUnconfident(container, data, message) {
    const node = templates.unconfident.content.cloneNode(true);
    $('[data-slot="message"]', node).textContent = message;
    buildBars($('[data-slot="bars"]', node), data.top3 || [], null);
    container.replaceChildren(node);
  }

  function renderError(container, message) {
    const node = templates.error.content.cloneNode(true);
    $('[data-slot="message"]', node).textContent = message;
    container.replaceChildren(node);
  }

  function setLoading(button, loading) {
    button.classList.toggle("is-loading", loading);
    button.disabled = loading;
    button.setAttribute("aria-busy", String(loading));
  }

  async function postJSON(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({ ok: false, error: "Unexpected server response." }));
    return data;
  }

  async function postForm(url, formData) {
    const response = await fetch(url, { method: "POST", body: formData });
    return response.json().catch(() => ({ ok: false, error: "Unexpected server response." }));
  }

  function updateCounter(textarea, counter) {
    counter.textContent = `${textarea.value.length} / ${textarea.maxLength}`;
  }

  [
    [$("#symptoms-text"), $("#symptoms-count")],
    [$("#sentiment-text"), $("#sentiment-count")],
  ].forEach(([textarea, counter]) => {
    textarea.addEventListener("input", () => updateCounter(textarea, counter));
  });

  $$(".example-chip").forEach((button) => {
    button.addEventListener("click", () => {
      const target = document.getElementById(button.dataset.target);
      target.value = button.dataset.value;
      target.dispatchEvent(new Event("input"));
      target.focus();
    });
  });

  // Symptoms ----------------------------------------------------------------
  const symptomsText = $("#symptoms-text");
  const symptomsBtn = $("#symptoms-submit");
  const symptomsResult = $("#symptoms-result");

  symptomsBtn.addEventListener("click", async () => {
    const text = symptomsText.value.trim();
    if (!text) return renderError(symptomsResult, "Add a symptom description before analysing.");
    setLoading(symptomsBtn, true);
    try {
      const data = await postJSON("/api/symptoms", { text });
      if (!data.ok) renderError(symptomsResult, data.error || "Something went wrong.");
      else if (data.confident) renderConfident(symptomsResult, data, { showInfo: true });
      else renderUnconfident(symptomsResult, data, "The model could not identify a condition confidently. Please speak with a healthcare professional.");
    } catch {
      renderError(symptomsResult, "Could not reach the server. Check that the Flask app is running.");
    } finally { setLoading(symptomsBtn, false); }
  });

  $("#symptoms-clear").addEventListener("click", () => {
    symptomsText.value = "";
    symptomsResult.innerHTML = "";
    updateCounter(symptomsText, $("#symptoms-count"));
    symptomsText.focus();
  });

  // Reusable image picker ---------------------------------------------------
  function createImagePicker({ input, dropzone, empty, preview, submit, clear, result, onSelect }) {
    let file = null;
    let objectUrl = null;

    function select(nextFile, { sample = false } = {}) {
      if (!nextFile?.type.startsWith("image/")) {
        renderError(result, "Please choose a JPG or PNG image.");
        return false;
      }
      if (nextFile.size > 8 * 1024 * 1024) {
        renderError(result, "That image is larger than 8 MB. Please choose a smaller file.");
        return false;
      }
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      file = nextFile;
      objectUrl = URL.createObjectURL(nextFile);
      preview.src = objectUrl;
      preview.hidden = false;
      empty.hidden = true;
      submit.disabled = false;
      clear.hidden = false;
      result.innerHTML = "";
      onSelect?.(file, sample);
      return true;
    }

    function reset() {
      file = null;
      input.value = "";
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = null;
      preview.removeAttribute("src");
      preview.hidden = true;
      empty.hidden = false;
      submit.disabled = true;
      clear.hidden = true;
      result.innerHTML = "";
      onSelect?.(null, false);
    }

    dropzone.addEventListener("click", () => input.click());
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); }
    });
    ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
      event.preventDefault(); dropzone.classList.add("dragover");
    }));
    ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
      event.preventDefault(); dropzone.classList.remove("dragover");
    }));
    dropzone.addEventListener("drop", (event) => select(event.dataTransfer.files?.[0]));
    input.addEventListener("change", () => select(input.files?.[0]));
    clear.addEventListener("click", reset);
    return { getFile: () => file, select, reset };
  }

  // Skin image --------------------------------------------------------------
  const imageSubmit = $("#image-submit");
  const imageResult = $("#image-result");
  const imagePicker = createImagePicker({
    input: $("#image-input"), dropzone: $("#image-dropzone"), empty: $("#dropzone-empty"),
    preview: $("#image-preview"), submit: imageSubmit, clear: $("#image-clear"), result: imageResult,
  });

  imageSubmit.addEventListener("click", async () => {
    const file = imagePicker.getFile();
    if (!file) return;
    setLoading(imageSubmit, true);
    try {
      const form = new FormData();
      form.append("image", file);
      const data = await postForm("/api/image", form);
      if (!data.ok) renderError(imageResult, data.error || "Something went wrong.");
      else if (data.invalid_image) renderError(imageResult, data.message);
      else if (data.confident) renderConfident(imageResult, data);
      else renderUnconfident(imageResult, data, "The model could not recognise a condition confidently. Please consult a dermatologist.");
    } catch { renderError(imageResult, "Could not reach the server. Check that the Flask app is running."); }
    finally { setLoading(imageSubmit, false); }
  });

  // Sentiment ---------------------------------------------------------------
  const sentimentText = $("#sentiment-text");
  const sentimentBtn = $("#sentiment-submit");
  const sentimentResult = $("#sentiment-result");
  const sentimentIcon = { Positive: "↗", Neutral: "—", Negative: "↘" };

  sentimentBtn.addEventListener("click", async () => {
    const text = sentimentText.value.trim();
    if (!text) return renderError(sentimentResult, "Add feedback text before analysing.");
    setLoading(sentimentBtn, true);
    try {
      const data = await postJSON("/api/sentiment", { text });
      if (!data.ok) renderError(sentimentResult, data.error || "Something went wrong.");
      else if (data.confident) {
        const result = renderConfident(sentimentResult, data);
        result.dataset.sentiment = data.prediction;
        $(".result-icon", result).textContent = sentimentIcon[data.prediction] || "•";
        $(".result-title", result).textContent = `Overall sentiment: ${data.prediction}`;
      } else renderUnconfident(sentimentResult, data, "The model could not classify this feedback confidently.");
    } catch { renderError(sentimentResult, "Could not reach the server. Check that the Flask app is running."); }
    finally { setLoading(sentimentBtn, false); }
  });

  $("#sentiment-clear").addEventListener("click", () => {
    sentimentText.value = "";
    sentimentResult.innerHTML = "";
    updateCounter(sentimentText, $("#sentiment-count"));
    sentimentText.focus();
  });

  // OCR ---------------------------------------------------------------------
  const ocrSubmit = $("#ocr-submit");
  const ocrResult = $("#ocr-result");
  const ocrPlaceholder = $("#ocr-placeholder");
  const ocrOutput = $("#ocr-output");
  const ocrText = $("#ocr-text");
  const sampleRibbon = $("#sample-ribbon");
  const ocrPicker = createImagePicker({
    input: $("#ocr-input"), dropzone: $("#ocr-dropzone"), empty: $("#ocr-dropzone-empty"),
    preview: $("#ocr-preview"), submit: ocrSubmit, clear: $("#ocr-clear"), result: ocrResult,
    onSelect: (file, sample) => {
      sampleRibbon.hidden = !sample;
      ocrOutput.hidden = true;
      ocrPlaceholder.hidden = false;
      if (!file) ocrText.value = "";
    },
  });

  $("#ocr-sample").addEventListener("click", async () => {
    const button = $("#ocr-sample");
    const sampleSelect = $("#ocr-sample-select");
    const sampleLabel = sampleSelect.options[sampleSelect.selectedIndex].text;
    const sampleUrl = sampleSelect.value;
    button.disabled = true;
    button.textContent = "Loading…";
    try {
      const response = await fetch(sampleUrl);
      if (!response.ok) throw new Error("Sample unavailable");
      const blob = await response.blob();
      const filename = sampleUrl.split("/").pop() || "ocr-test-sample.png";
      const file = new File([blob], filename, { type: blob.type || "image/png" });
      ocrPicker.select(file, { sample: true });
      showToast(`${sampleLabel} sample ready`);
    } catch { renderError(ocrResult, "The demo image could not be loaded."); }
    finally { button.disabled = false; button.textContent = "Load sample"; }
  });

  ocrSubmit.addEventListener("click", async () => {
    const file = ocrPicker.getFile();
    if (!file) return;
    setLoading(ocrSubmit, true);
    ocrResult.innerHTML = "";
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("mode", $('input[name="ocr-mode"]:checked').value);
      const data = await postForm("/api/ocr", form);
      if (!data.ok) return renderError(ocrResult, data.error || "OCR could not read this image.");
      if (!data.text) return renderError(ocrResult, data.message || "No readable text was found.");
      ocrText.value = data.text;
      $("#ocr-stats").textContent = `${data.word_count} words · ${data.character_count} characters`;
      $("#ocr-confidence").textContent = `${Math.round(data.confidence)}% OCR quality`;
      ocrPlaceholder.hidden = true;
      ocrOutput.hidden = false;
      showToast("Text extracted successfully");
    } catch { renderError(ocrResult, "Could not reach the OCR service. Check that the Flask app is running."); }
    finally { setLoading(ocrSubmit, false); }
  });

  $("#ocr-copy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(ocrText.value); }
    catch { ocrText.select(); document.execCommand("copy"); }
    showToast("Extracted text copied");
  });

  $("#ocr-download").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([ocrText.value], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "carelens-ocr-text.txt";
    link.click();
    URL.revokeObjectURL(url);
    showToast("Text file downloaded");
  });

  $("#ocr-send-symptoms").addEventListener("click", () => {
    symptomsText.value = ocrText.value.slice(0, symptomsText.maxLength);
    updateCounter(symptomsText, $("#symptoms-count"));
    activateTab("symptoms", true);
    symptomsText.focus();
    showToast("Extracted text added to symptoms");
  });
})();
