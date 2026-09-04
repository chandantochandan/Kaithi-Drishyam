const input = document.querySelector("#documentInput");
const canvas = document.querySelector("#previewCanvas");
const emptyState = document.querySelector("#emptyState");
const fileName = document.querySelector("#fileName");
const ctx = canvas.getContext("2d");

function drawPlaceholder() {
  ctx.fillStyle = "#f3f1ea";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
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

    emptyState.hidden = true;
    URL.revokeObjectURL(image.src);
  };
  image.src = URL.createObjectURL(file);
}

input.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (!file) {
    fileName.textContent = "Waiting for upload";
    emptyState.hidden = false;
    drawPlaceholder();
    return;
  }

  fileName.textContent = file.name;
  drawImage(file);
});

drawPlaceholder();
