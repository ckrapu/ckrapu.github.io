(() => {
  const canvas = document.querySelector("[data-ising-sampler]");
  if (!canvas || !canvas.getContext) return;

  const context = canvas.getContext("2d", { alpha: true });
  const visualContainer = canvas.closest(".about-intro__visual");
  const stackedLayout = window.matchMedia("(max-width: 720px)");
  const isometricLayout = window.matchMedia("(min-width: 721px)");
  const rows = 56;
  const targetPointSpacing = 4.5;
  const maximumTemperature = 3;
  const temperatureSweepRate = 0.75;
  const activityDecayMs = 1150;
  const idleActivity = 0.015;
  const updatesPerSecond = 1.9;
  const fadeDurationMs = 115;
  const pointerRadius = 2;

  let columns = 0;
  let spins = new Int8Array();
  let displaySpins = new Float32Array();
  let projectedX = new Float32Array();
  let projectedY = new Float32Array();
  let cssWidth = 0;
  let cssHeight = 0;
  let pixelRatio = 1;
  let lastActivity = performance.now();
  let lastFrame = performance.now();
  let updateRemainder = 0;
  let frameRequest = 0;
  let isVisible = true;
  let needsPaint = true;
  let lastPointerCell = -1;
  let temperature = 1 / 0.48;
  let temperatureDirection = 1;
  let isTemperatureHeld = false;

  const indexFor = (row, column) => row * columns + column;

  function gibbsUpdate() {
    const index = Math.floor(Math.random() * spins.length);
    const row = Math.floor(index / columns);
    const column = index % columns;
    const left = indexFor(row, (column + columns - 1) % columns);
    const right = indexFor(row, (column + 1) % columns);
    const above = indexFor((row + rows - 1) % rows, column);
    const below = indexFor((row + 1) % rows, column);
    const neighborSum = spins[left] + spins[right] + spins[above] + spins[below];
    const probabilityOn =
      temperature <= 0.001
        ? neighborSum > 0
          ? 1
          : neighborSum < 0
            ? 0
            : 0.5
        : 1 / (1 + Math.exp((-2 * neighborSum) / temperature));
    spins[index] = Math.random() < probabilityOn ? 1 : -1;
  }

  function seedField() {
    spins = new Int8Array(rows * columns);
    for (let index = 0; index < spins.length; index += 1) {
      spins[index] = Math.random() < 0.5 ? -1 : 1;
    }

    displaySpins = new Float32Array(spins.length);
    projectedX = new Float32Array(spins.length);
    projectedY = new Float32Array(spins.length);
    for (let index = 0; index < spins.length; index += 1) {
      displaySpins[index] = spins[index] === 1 ? 1 : 0;
    }
  }

  function resize() {
    const nextWidth = canvas.getBoundingClientRect().width;
    if (!nextWidth) return;

    const nextColumns = Math.max(72, Math.floor(nextWidth / targetPointSpacing));
    cssWidth = nextWidth;
    cssHeight = stackedLayout.matches
      ? Math.max(92, Math.min(132, nextWidth * 0.145))
      : Math.max(280, visualContainer?.getBoundingClientRect().height || 0);
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);

    canvas.style.height = `${cssHeight}px`;
    canvas.width = Math.round(cssWidth * pixelRatio);
    canvas.height = Math.round(cssHeight * pixelRatio);
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    if (nextColumns !== columns) {
      columns = nextColumns;
      seedField();
    }

    needsPaint = true;
    draw();
  }

  function draw() {
    if (!cssWidth || !spins.length) return;

    const styles = getComputedStyle(canvas);
    const onColor = styles.getPropertyValue("--global-theme-color").trim() || "#018406";

    context.clearRect(0, 0, cssWidth, cssHeight);
    context.fillStyle = onColor;
    context.shadowColor = onColor;
    context.shadowBlur = 2.25;

    for (let row = 0; row < rows; row += 1) {
      const depth = row / (rows - 1);
      const depthFromCenter = depth - 0.5;
      const isIsometric = isometricLayout.matches;
      const perspective = 0.68 + depth * 0.32;
      const verticalFade = Math.min(1, depth / 0.09) * Math.min(1, (1 - depth) / 0.1);

      for (let column = 0; column < columns; column += 1) {
        const index = indexFor(row, column);
        const left = indexFor(row, (column + columns - 1) % columns);
        const right = indexFor(row, (column + 1) % columns);
        const above = indexFor((row + rows - 1) % rows, column);
        const below = indexFor((row + 1) % rows, column);
        const localField =
          (displaySpins[index] * 4 + displaySpins[left] + displaySpins[right] + displaySpins[above] + displaySpins[below]) / 8;
        const horizontalPhase = (column / columns) * Math.PI * 2;
        const undulation = Math.sin(horizontalPhase * 3 + row * 0.17) * 0.035 + Math.sin(horizontalPhase - row * 0.29) * 0.025;
        const elevation = Math.max(0, Math.min(1, 0.08 + localField * 0.72 + undulation));
        const worldX = column / (columns - 1) - 0.5;
        const x = isIsometric
          ? cssWidth * 0.5 + (worldX + depthFromCenter) * cssWidth * 0.46
          : cssWidth * 0.5 + worldX * cssWidth * 1.06 * perspective;
        const floorY = isIsometric
          ? cssHeight * 0.56 + (depthFromCenter - worldX) * cssHeight * 0.27
          : cssHeight * (0.035 + depth * 0.93);
        const projectionScale = isIsometric ? 0.9 : perspective;
        const y = floorY - elevation * cssHeight * 0.24 * projectionScale;

        projectedX[index] = x;
        projectedY[index] = y;

        if (displaySpins[index] <= 0.002) continue;

        const pointSize = 0.55 + projectionScale * 1.15;
        context.globalAlpha = displaySpins[index] * verticalFade * (0.42 + projectionScale * 0.58);
        context.fillRect(x - pointSize * 0.5, y - pointSize * 0.5, pointSize, pointSize);
      }
    }
    context.globalAlpha = 1;
    context.shadowBlur = 0;
    needsPaint = false;
  }

  function animate(now) {
    frameRequest = 0;
    if (!isVisible) return;

    const elapsed = Math.min(now - lastFrame, 50);
    lastFrame = now;

    if (isTemperatureHeld) {
      lastActivity = now;
      temperature += temperatureDirection * temperatureSweepRate * (elapsed / 1000);

      if (temperature >= maximumTemperature) {
        temperature = maximumTemperature - (temperature - maximumTemperature);
        temperatureDirection = -1;
      } else if (temperature <= 0) {
        temperature = -temperature;
        temperatureDirection = 1;
      }
    }

    const recentActivity = Math.exp(-(now - lastActivity) / activityDecayMs);
    const activity = idleActivity + (1 - idleActivity) * recentActivity;
    updateRemainder += (spins.length * updatesPerSecond * activity * elapsed) / 1000;
    const updates = Math.floor(updateRemainder);
    updateRemainder -= updates;
    for (let index = 0; index < updates; index += 1) gibbsUpdate();

    let isFading = false;
    const fadeAmount = 1 - Math.exp(-elapsed / fadeDurationMs);
    for (let index = 0; index < spins.length; index += 1) {
      const target = spins[index] === 1 ? 1 : 0;
      const difference = target - displaySpins[index];
      if (Math.abs(difference) > 0.004) {
        displaySpins[index] += difference * fadeAmount;
        isFading = true;
      } else if (displaySpins[index] !== target) {
        displaySpins[index] = target;
        needsPaint = true;
      }
    }

    if (isFading || needsPaint) draw();
    frameRequest = requestAnimationFrame(animate);
  }

  function wake() {
    lastActivity = performance.now();
    if (!isVisible || frameRequest) return;
    lastFrame = lastActivity;
    frameRequest = requestAnimationFrame(animate);
  }

  function flipPointerCell(event) {
    const bounds = canvas.getBoundingClientRect();
    const pointerX = event.clientX - bounds.left;
    const pointerY = event.clientY - bounds.top;
    let index = -1;
    let nearestDistance = Infinity;

    for (let candidate = 0; candidate < projectedX.length; candidate += 1) {
      const xDistance = pointerX - projectedX[candidate];
      const yDistance = pointerY - projectedY[candidate];
      const distance = xDistance * xDistance + yDistance * yDistance;
      if (distance < nearestDistance) {
        nearestDistance = distance;
        index = candidate;
      }
    }

    if (index < 0 || nearestDistance > 144) return;
    if (index === lastPointerCell) return;

    const row = Math.floor(index / columns);
    const column = index % columns;

    for (let rowOffset = -pointerRadius; rowOffset <= pointerRadius; rowOffset += 1) {
      for (let columnOffset = -pointerRadius; columnOffset <= pointerRadius; columnOffset += 1) {
        if (rowOffset ** 2 + columnOffset ** 2 > pointerRadius ** 2) continue;

        const affectedRow = row + rowOffset;
        const affectedColumn = (column + columnOffset + columns) % columns;
        if (affectedRow < 0 || affectedRow >= rows) continue;

        spins[indexFor(affectedRow, affectedColumn)] *= -1;
      }
    }

    lastPointerCell = index;
    needsPaint = true;
    wake();
  }

  canvas.addEventListener("pointermove", flipPointerCell, { passive: true });
  canvas.addEventListener("pointerdown", (event) => {
    isTemperatureHeld = true;
    canvas.setPointerCapture?.(event.pointerId);
    wake();
  });
  canvas.addEventListener("pointerleave", () => {
    lastPointerCell = -1;
  });

  ["pointerup", "pointercancel"].forEach((eventName) => {
    window.addEventListener(eventName, () => {
      isTemperatureHeld = false;
    });
  });

  ["pointermove", "pointerdown", "keydown", "scroll", "touchmove"].forEach((eventName) => {
    window.addEventListener(eventName, wake, { passive: true });
  });

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(canvas);
  if (visualContainer) resizeObserver.observe(visualContainer);

  new MutationObserver(() => {
    needsPaint = true;
    draw();
  }).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme", "data-theme-setting"],
  });

  new IntersectionObserver(([entry]) => {
    isVisible = entry.isIntersecting;
    if (isVisible) wake();
  }).observe(canvas);

  isometricLayout.addEventListener("change", () => {
    needsPaint = true;
    draw();
  });

  resize();
  frameRequest = requestAnimationFrame(animate);
})();
