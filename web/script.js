const input = document.querySelector("#documentInput");
const canvas = document.querySelector("#previewCanvas");
const emptyState = document.querySelector("#emptyState");
const fileName = document.querySelector("#fileName");
const preprocessingStatus = document.querySelector("#preprocessingStatus");
const segmentationStatus = document.querySelector("#segmentationStatus");
const recognitionStatus = document.querySelector("#recognitionStatus");
const bhashiniStatus = document.querySelector("#bhashiniStatus");
const lineSummary = document.querySelector("#lineSummary");
const lineList = document.querySelector("#lineList");
const ctx = canvas.getContext("2d");
const apiBaseUrl = window.KAITHI_API_BASE_URL || "http://127.0.0.1:8001";

function drawPlaceholder() {
  ctx.fillStyle = "#f3f1ea";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function setStatus(message) {
  preprocessingStatus.textContent = message;
  segmentationStatus.textContent = message;
  lineSummary.hidden = true;
  lineList.replaceChildren();
}

function drawImage(file) {
  const image = new Image();
  image.onload = () => {
    const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
    const width = image.width * scale;
    const height = image.height * scale;
    const x = (canvas.width - width) / 2;
    const y = (canvas.height - height) / 2;

    ctx.fillStyle = "#f3f1ea";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, x, y, width, height);

    ctx.strokeStyle = "#2f6f73";
    ctx.lineWidth = 3;
    ctx.strokeRect(x + 18, y + 26, Math.max(40, width - 36), Math.max(40, height - 52));

    emptyState.classList.add("is-hidden");
    URL.revokeObjectURL(image.src);
  };
  image.src = URL.createObjectURL(file);
}

function renderResult(payload) {
  const preprocessing = payload.preprocessing;
  const segmentation = payload.segmentation;
  const recognition = payload.recognition;
  const transliteration = payload.transliteration;

  preprocessingStatus.textContent = [
    `${preprocessing.metadata.steps_applied.length} steps`,
    `deskew ${preprocessing.deskew_angle.toFixed(2)} deg`,
    `noise ${preprocessing.noise_level.toFixed(3)}`,
  ].join(" | ");
  segmentationStatus.textContent = `${segmentation.line_count} text line(s) detected on ${payload.page_count || 1} page(s)`;
  recognitionStatus.textContent = recognition.message;
  if (transliteration.status === "transliterated") {
    recognitionStatus.textContent = `${recognition.message} Devanagari: ${transliteration.devanagari_text}`;
  }

  if (segmentation.lines.length > 0) {
    lineSummary.hidden = false;
    lineList.replaceChildren(
      ...segmentation.lines.slice(0, 8).map((line) => {
        const item = document.createElement("li");
        const box = line.bounding_box;
        item.textContent = `Line ${line.reading_order + 1}: x ${box.x}, y ${box.y}, ${box.width}x${box.height}`;
        return item;
      }),
    );
  } else {
    lineSummary.hidden = false;
    const item = document.createElement("li");
    item.textContent = "No line boxes detected for this image. Try a clearer, flatter scan.";
    lineList.replaceChildren(item);
  }
}

async function processDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${apiBaseUrl}/api/v1/documents/process`, {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    const message = typeof payload.detail === "string" ? payload.detail : payload.detail?.message;
    throw new Error(message || "Document processing failed");
  }

  return payload;
}

input.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) {
    fileName.textContent = "Waiting for upload";
    emptyState.classList.remove("is-hidden");
    setStatus("Waiting for upload");
    drawPlaceholder();
    return;
  }

  fileName.textContent = file.name;
  setStatus("Processing...");
  recognitionStatus.textContent = "Pending model training";
  bhashiniStatus.textContent = "Checking local pipeline";
  drawImage(file);

  try {
    const payload = await processDocument(file);
    renderResult(payload);
    bhashiniStatus.textContent = "Local fallback used unless KAITHI_BHASHINI_API_KEY is configured";
  } catch (error) {
    preprocessingStatus.textContent = "Failed";
    segmentationStatus.textContent = error.message;
    bhashiniStatus.textContent = "Backend not reachable or request failed";
  }
});

drawPlaceholder();
