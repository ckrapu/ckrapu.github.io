---
title: Using Antigravity for Statistical Physics in JS
layout: post
date: 2025-11-21
description: Testing Google's new IDE and top model on a ferromagnetic simulation
---

I like learning about the hidden benchmarks that everyone seems to bring out when a new large language model drops. Mine used to be asking the model about obscure but well-documented people on the internet like family members or acquaintances in the sciences or with IMDB credits. Since ~late 2024, most models are nailing that one so it's not as interesting. Instead, I've moved onto Javascript-based visualizations. of statistical physics

Since Gemini 3 and Google's Antigravity IDE were released recently (and yes, I am aware it is basically Windsurf), I wanted to give it a try with an easy one - the Ising model of ferromagnetism.

Here's what Antigravity with Gemini 3 Pro cooked up in an hour:

<style>
    :root {
        --bg-color: #1c1c1d;
        --text-color: #ffffff;
        --accent-color: #00ffcc;
        --secondary-color: #ff00ff;
        --grid-gap: 2px;
        --control-bg: rgba(255, 255, 255, 0.1);
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }

    #ising-app.light-theme {
        --bg-color: #ffffff;
        --text-color: #333333;
        --accent-color: #007bff;
        --secondary-color: #dc3545;
        --control-bg: rgba(0, 0, 0, 0.05);
    }

    #ising-app {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: var(--font-family);
        margin: 2rem 0;
        padding: 1.5rem 1rem;
        border-radius: 12px;
        transition: background-color 0.3s, color 0.3s;
    }

    h1 {
        font-weight: 300;
        letter-spacing: 2px;
        margin-bottom: 10px;
        text-transform: uppercase;
        font-size: 1.5rem;
    }

    #main-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 20px;
        padding-bottom: 40px;
    }

    #canvas-container {
        position: relative;
        box-shadow: 0 0 50px rgba(0, 0, 0, 0.5);
        border-radius: 12px;
        overflow: hidden;
        cursor: none;
    }

    canvas {
        display: block;
    }

    #info-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.85);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        color: #fff;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 40px;
        box-sizing: border-box;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.3s ease, visibility 0.3s ease;
        z-index: 10;
        text-align: center;
        pointer-events: none;
    }

    #ising-app.light-theme #info-overlay {
        background: rgba(255, 255, 255, 0.9);
        color: #333;
    }

    #info-overlay.visible {
        opacity: 1;
        visibility: visible;
        pointer-events: auto;
    }

    #info-content {
        max-width: 500px;
    }

    #info-content h2 {
        margin-top: 0;
        font-weight: 300;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 20px;
        border-bottom: 1px solid var(--accent-color);
        display: inline-block;
        padding-bottom: 5px;
    }

    #info-content p {
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 15px;
        opacity: 0.9;
    }

    .math-display {
        font-family: "Times New Roman", Times, serif;
        font-style: italic;
        font-size: 1.2rem;
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: block;
    }

    #ising-app.light-theme .math-display {
        background: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    .math-inline {
        font-family: "Times New Roman", Times, serif;
        font-style: italic;
        padding: 0 2px;
    }

    #controls {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
        background: var(--control-bg);
        padding: 15px 30px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
    }

    #controls-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 5px;
    }

    #controls-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
        line-height: 1;
    }

    #info-icon {
        width: 16px;
        height: 16px;
        cursor: help;
        opacity: 0.7;
        transition: opacity 0.2s;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    #info-icon:hover {
        opacity: 1;
    }

    #info-icon svg {
        width: 100%;
        height: 100%;
        fill: currentColor;
    }

    .math {
        font-family: 'Times New Roman', Times, serif;
        font-style: italic;
        background: rgba(255, 255, 255, 0.1);
        padding: 2px 5px;
        border-radius: 4px;
        display: block;
        text-align: center;
        margin: 10px 0;
    }

    #ising-app.light-theme .math {
        background: rgba(0, 0, 0, 0.05);
    }

    #controls-body {
        display: flex;
        gap: 20px;
        align-items: center;
    }

    .control-group {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 5px;
    }

    label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.8;
    }

    input[type="range"] {
        -webkit-appearance: none;
        appearance: none;
        width: 100px;
        height: 4px;
        background: rgba(128, 128, 128, 0.3);
        border-radius: 2px;
        outline: none;
    }

    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 14px;
        height: 14px;
        background: var(--text-color);
        border-radius: 50%;
        cursor: pointer;
        transition: transform 0.1s;
    }

    input[type="range"]::-webkit-slider-thumb:hover {
        transform: scale(1.2);
    }

    .icon-btn {
        background: none;
        border: 1px solid var(--text-color);
        color: var(--text-color);
        width: 36px;
        height: 36px;
        border-radius: 50%;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        transition: all 0.2s;
    }

    .icon-btn:hover {
        background: var(--text-color);
        color: var(--bg-color);
    }

    .icon-btn svg {
        width: 18px;
        height: 18px;
        fill: currentColor;
    }

    #plot-container {
        width: 600px;
        max-width: 100%;
        height: 60px;
        background: transparent;
        border-radius: 8px;
        overflow: hidden;
        position: relative;
        margin-top: -10px;
    }

    #plotCanvas {
        width: 100%;
        height: 100%;
        cursor: default;
    }

    .value-display {
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>

<div id="ising-app">
    <div id="main-container">
        <div id="canvas-container">
            <canvas id="simCanvas" width="600" height="600"></canvas>

            <div id="info-overlay">
                <div id="info-content">
                    <h2>The Ising Model</h2>
                    <p>
                        A mathematical model of ferromagnetism in statistical mechanics. The grid consists of discrete
                        variables (spins) that can be in one of two states (+1 or -1).
                    </p>
                    <div class="math-display">
                        H(&sigma;) = -J &sum;<sub>&lt;ij&gt;</sub> &sigma;<sub>i</sub>&sigma;<sub>j</sub> - h
                        &sum;<sub>j</sub> &sigma;<sub>j</sub>
                    </div>
                    <p>
                        <strong>Simulation:</strong> This visualization uses a <em>Random Scan Gibbs Sampler</em>. In
                        each step, a single spin is chosen at random and updated based on the Boltzmann distribution
                        determined by its neighbors and the external field.
                    </p>
                    <p style="font-size: 0.85rem; opacity: 0.7; margin-top: 20px;">
                        Named after physicist Ernst Ising, who solved the 1D model in his 1924 thesis.
                    </p>
                </div>
            </div>
        </div>

        <div id="controls">
            <div id="controls-header">
                <span id="controls-title">Ising Model</span>
                <div id="info-icon">
                    <svg viewBox="0 0 24 24">
                        <path
                            d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
                    </svg>
                </div>
            </div>

            <div id="controls-body">
                <button id="theme-toggle" class="icon-btn" title="Toggle Theme">
                    <svg class="sun-icon" viewBox="0 0 24 24" style="display: none;">
                        <path
                            d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41L5.99 4.58zm12.37 12.37c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0 .39-.39.39-1.03 0-1.41l-1.06-1.06zm1.06-10.96c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06zM7.05 18.36c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06z" />
                    </svg>
                    <svg class="moon-icon" viewBox="0 0 24 24">
                        <path
                            d="M9.37 5.51c-.18.64-.27 1.31-.27 1.99 0 4.08 3.32 7.4 7.4 7.4.68 0 1.35-.09 1.99-.27C17.45 18.55 14.4 21 10.75 21c-4.83 0-8.75-3.92-8.75-8.75 0-3.65 2.45-6.7 5.37-7.74z" />
                    </svg>
                </button>

                <div class="control-group">
                    <label>Temperature <span id="temp-val" class="value-display">2.27</span></label>
                    <input type="range" id="temp-slider" min="0.1" max="5.0" step="0.01" value="2.27">
                </div>

                <div class="control-group">
                    <label>External Field <span id="field-val" class="value-display">0.00</span></label>
                    <input type="range" id="field-slider" min="-2.0" max="2.0" step="0.1" value="0.0">
                </div>

                <div class="control-group">
                    <label>Speed <span id="speed-val" class="value-display">4</span></label>
                    <input type="range" id="speed-slider" min="1" max="10" step="1" value="4">
                </div>

                <button id="restart-btn" class="icon-btn" title="Restart Demo">
                    <svg viewBox="0 0 24 24">
                        <path
                            d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" />
                    </svg>
                </button>
            </div>
        </div>

        <div id="plot-container">
            <canvas id="plotCanvas" width="600" height="60"></canvas>
        </div>
    </div>
</div>

<script>
    // Configuration
    const GRID_SIZE = 30;
    const CELL_SIZE = 20;
    const J = 1;
    const BRUSH_RADIUS = 1.5 * CELL_SIZE;

    // App root
    const app = document.getElementById('ising-app');

    // State
    let grid = [];
    let temperature = 2.27;
    let externalField = 0.0;
    let speed = 4;
    let isMouseDown = false;
    let isDarkTheme = true;
    let mouseX = -1000;
    let mouseY = -1000;

    let isDemoMode = true;
    let demoTime = 0;

    const MAX_HISTORY = 300;
    let magnetizationHistory = new Array(MAX_HISTORY).fill(0);

    // DOM Elements
    const canvas = document.getElementById('simCanvas');
    const ctx = canvas.getContext('2d');
    const plotCanvas = document.getElementById('plotCanvas');
    const plotCtx = plotCanvas.getContext('2d');

    const tempSlider = document.getElementById('temp-slider');
    const tempVal = document.getElementById('temp-val');
    const fieldSlider = document.getElementById('field-slider');
    const fieldVal = document.getElementById('field-val');
    const speedSlider = document.getElementById('speed-slider');
    const speedVal = document.getElementById('speed-val');
    const themeToggle = document.getElementById('theme-toggle');
    const restartBtn = document.getElementById('restart-btn');

    const infoIcon = document.getElementById('info-icon');
    const infoOverlay = document.getElementById('info-overlay');

    function initGrid() {
        grid = [];
        for (let i = 0; i < GRID_SIZE; i++) {
            let row = [];
            for (let j = 0; j < GRID_SIZE; j++) {
                row.push(Math.random() > 0.5 ? 1 : -1);
            }
            grid.push(row);
        }
    }

    function resetDemo() {
        isDemoMode = true;
        demoTime = 0;
        initGrid();
        magnetizationHistory = new Array(MAX_HISTORY).fill(0);
    }

    function draw() {
        const bgColor = getComputedStyle(app).getPropertyValue('--bg-color').trim();
        const accentColor = getComputedStyle(app).getPropertyValue('--accent-color').trim();
        const secondaryColor = getComputedStyle(app).getPropertyValue('--secondary-color').trim();

        ctx.fillStyle = bgColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        for (let i = 0; i < GRID_SIZE; i++) {
            for (let j = 0; j < GRID_SIZE; j++) {
                const x = j * CELL_SIZE + CELL_SIZE / 2;
                const y = i * CELL_SIZE + CELL_SIZE / 2;
                const radius = (CELL_SIZE / 2) * 0.8;

                ctx.beginPath();
                ctx.arc(x, y, radius, 0, Math.PI * 2);

                if (grid[i][j] === 1) {
                    ctx.fillStyle = accentColor;
                    if (isDarkTheme) {
                        ctx.shadowBlur = 10;
                        ctx.shadowColor = accentColor;
                    } else {
                        ctx.shadowBlur = 0;
                    }
                } else {
                    ctx.fillStyle = secondaryColor;
                    if (isDarkTheme) {
                        ctx.shadowBlur = 10;
                        ctx.shadowColor = secondaryColor;
                    } else {
                        ctx.shadowBlur = 0;
                    }
                }

                ctx.fill();
                ctx.shadowBlur = 0;
            }
        }

        if (isMouseDown) {
            ctx.beginPath();
            ctx.arc(mouseX, mouseY, BRUSH_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(128, 128, 128, 0.55)';
            ctx.fill();
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }

    function drawPlot() {
        plotCtx.clearRect(0, 0, plotCanvas.width, plotCanvas.height);

        const accentColor = getComputedStyle(app).getPropertyValue('--accent-color').trim();
        const secondaryColor = getComputedStyle(app).getPropertyValue('--secondary-color').trim();
        const textColor = getComputedStyle(app).getPropertyValue('--text-color').trim();

        plotCtx.lineWidth = 2;
        plotCtx.beginPath();

        for (let i = 0; i < MAX_HISTORY; i++) {
            const x = (i / (MAX_HISTORY - 1)) * plotCanvas.width;
            const m = magnetizationHistory[i];
            const y = ((1 - m) / 2) * plotCanvas.height;

            if (i === 0) {
                plotCtx.moveTo(x, y);
            } else {
                plotCtx.lineTo(x, y);
            }
        }

        const gradient = plotCtx.createLinearGradient(0, plotCanvas.height, 0, 0);
        gradient.addColorStop(0, secondaryColor);
        gradient.addColorStop(1, accentColor);
        plotCtx.strokeStyle = gradient;
        plotCtx.stroke();

        plotCtx.fillStyle = textColor;
        plotCtx.font = '10px sans-serif';
        plotCtx.textAlign = 'center';
        plotCtx.textBaseline = 'bottom';
        plotCtx.globalAlpha = 0.7;
        plotCtx.fillText('Magnetization', plotCanvas.width / 2, plotCanvas.height - 2);
        plotCtx.globalAlpha = 1.0;
    }

    function update() {
        const updatesPerFrame = Math.floor(Math.pow(speed, 3) + 10);

        for (let k = 0; k < updatesPerFrame; k++) {
            const i = Math.floor(Math.random() * GRID_SIZE);
            const j = Math.floor(Math.random() * GRID_SIZE);

            const top = grid[(i - 1 + GRID_SIZE) % GRID_SIZE][j];
            const bottom = grid[(i + 1) % GRID_SIZE][j];
            const left = grid[i][(j - 1 + GRID_SIZE) % GRID_SIZE];
            const right = grid[i][(j + 1) % GRID_SIZE];

            const sumNeighbors = top + bottom + left + right;

            const beta = 1 / temperature;
            const effectiveField = J * sumNeighbors + externalField;
            const pUp = 1 / (1 + Math.exp(-2 * beta * effectiveField));

            grid[i][j] = Math.random() < pUp ? 1 : -1;
        }
    }

    function calculateMagnetization() {
        let sum = 0;
        for (let i = 0; i < GRID_SIZE; i++) {
            for (let j = 0; j < GRID_SIZE; j++) {
                sum += grid[i][j];
            }
        }
        return sum / (GRID_SIZE * GRID_SIZE);
    }

    function updateDemo() {
        if (!isDemoMode) return;

        demoTime += 0.01;

        temperature = 2.27 + 1.0 * Math.sin(demoTime * 0.5);
        externalField = 1.5 * Math.sin(demoTime * 0.8);

        tempSlider.value = temperature;
        tempVal.textContent = temperature.toFixed(2);

        fieldSlider.value = externalField;
        fieldVal.textContent = externalField.toFixed(2);
    }

    function loop() {
        updateDemo();
        update();
        draw();

        const m = calculateMagnetization();
        magnetizationHistory.push(m);
        magnetizationHistory.shift();
        drawPlot();

        requestAnimationFrame(loop);
    }

    function handleInteraction(e) {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        mouseX = x;
        mouseY = y;

        if (!isMouseDown) return;

        isDemoMode = false;

        for (let i = 0; i < GRID_SIZE; i++) {
            for (let j = 0; j < GRID_SIZE; j++) {
                const cellX = j * CELL_SIZE + CELL_SIZE / 2;
                const cellY = i * CELL_SIZE + CELL_SIZE / 2;

                const dist = Math.sqrt((x - cellX) ** 2 + (y - cellY) ** 2);

                if (dist < BRUSH_RADIUS) {
                    if (paintState !== 0) {
                        grid[i][j] = paintState;
                    }
                }
            }
        }
    }

    let paintState = 0;

    canvas.addEventListener('mousedown', (e) => {
        isMouseDown = true;
        isDemoMode = false;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const j = Math.floor(x / (rect.width / GRID_SIZE));
        const i = Math.floor(y / (rect.height / GRID_SIZE));

        if (i >= 0 && i < GRID_SIZE && j >= 0 && j < GRID_SIZE) {
            paintState = -grid[i][j];
        } else {
            paintState = 1;
        }

        handleInteraction(e);
    });

    window.addEventListener('mouseup', () => {
        isMouseDown = false;
        paintState = 0;
    });

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = e.clientY - rect.top;
        handleInteraction(e);
    });

    canvas.addEventListener('touchstart', (e) => {
        isMouseDown = true;
        isDemoMode = false;
        e.preventDefault();

        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        const j = Math.floor(x / (rect.width / GRID_SIZE));
        const i = Math.floor(y / (rect.height / GRID_SIZE));

        if (i >= 0 && i < GRID_SIZE && j >= 0 && j < GRID_SIZE) {
            paintState = -grid[i][j];
        } else {
            paintState = 1;
        }

        mouseX = x;
        mouseY = y;
        handleInteraction(touch);
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        mouseX = touch.clientX - rect.left;
        mouseY = touch.clientY - rect.top;
        handleInteraction(touch);
    }, { passive: false });

    window.addEventListener('touchend', () => {
        isMouseDown = false;
        paintState = 0;
    });

    function stopDemo() {
        isDemoMode = false;
    }

    tempSlider.addEventListener('input', (e) => {
        stopDemo();
        temperature = parseFloat(e.target.value);
        tempVal.textContent = temperature.toFixed(2);
    });

    fieldSlider.addEventListener('input', (e) => {
        stopDemo();
        externalField = parseFloat(e.target.value);
        fieldVal.textContent = externalField.toFixed(2);
    });

    speedSlider.addEventListener('input', (e) => {
        stopDemo();
        speed = parseInt(e.target.value);
        speedVal.textContent = speed;
    });

    themeToggle.addEventListener('click', () => {
        isDarkTheme = !isDarkTheme;
        const sunIcon = themeToggle.querySelector('.sun-icon');
        const moonIcon = themeToggle.querySelector('.moon-icon');

        if (isDarkTheme) {
            app.classList.remove('light-theme');
            sunIcon.style.display = 'none';
            moonIcon.style.display = 'block';
        } else {
            app.classList.add('light-theme');
            sunIcon.style.display = 'block';
            moonIcon.style.display = 'none';
        }
    });

    restartBtn.addEventListener('click', () => {
        resetDemo();
    });

    infoIcon.addEventListener('mouseenter', () => {
        infoOverlay.classList.add('visible');
    });

    infoIcon.addEventListener('mouseleave', () => {
        infoOverlay.classList.remove('visible');
    });

    infoOverlay.addEventListener('mouseenter', () => {
        infoOverlay.classList.add('visible');
    });

    infoOverlay.addEventListener('mouseleave', () => {
        infoOverlay.classList.remove('visible');
    });

    initGrid();
    loop();
</script>

I've gotten decent results out of the last crop of OpenAI and Anthropic models, but the Chrome browser extension for retrieving the DOM really helped too. It's a great feature, and I expect Cursor to have something similar soon.  I think some of the other UI features like showing subtasks and intermediate steps were a little unnecessary. Overall, great work by the former Windsurf team and the other G staff!