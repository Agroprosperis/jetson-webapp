(() => {
  const panel = document.getElementById("stagePanel");
  if (!panel) return;

  const canMove = panel.dataset.canMove === "true";
  const noteEl = document.getElementById("stageNote");
  const posEl = document.getElementById("stagePos");
  const voltEl = document.getElementById("stageVolt");
  const speedEl = document.getElementById("stageSpeed");
  const stepEl = document.getElementById("stageStep");
  const pointsEl = document.getElementById("stagePoints");
  const routesEl = document.getElementById("stageRoutes");
  const nameEl = document.getElementById("stageRouteName");
  const canvas = document.getElementById("stageMap");
  const box = ["minX", "maxX", "minY", "maxY"].reduce((acc, key) => {
    acc[key] = document.getElementById("stage" + key[0].toUpperCase() + key.slice(1));
    return acc;
  }, {});

  let state = null;

  function setNote(text, isError) {
    noteEl.textContent = text || "";
    noteEl.style.color = isError ? "#fca5a5" : "#bbb";
  }

  async function send(path, options) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      setNote(body.error || "Помилка столу", true);
      return null;
    }
    apply(body);
    return body;
  }

  function apply(next) {
    state = next;
    const pos = next.position_mm || { x: 0, y: 0 };
    posEl.textContent = "X " + pos.x.toFixed(2) + "   Y " + pos.y.toFixed(2);
    if (next.voltage_v) {
      voltEl.textContent = "Живлення X " + next.voltage_v.x.toFixed(1) + " В, Y " + next.voltage_v.y.toFixed(1) + " В";
    } else if (next.simulated) {
      voltEl.textContent = "Стіл не підключений до цього комп'ютера";
    } else {
      voltEl.textContent = "Немає відповіді від столу";
    }
    if (document.activeElement !== speedEl) speedEl.value = next.speed_mm_s;
    if (document.activeElement !== stepEl) stepEl.value = String(next.step_mm);
    const workspace = next.workspace_mm || {};
    if (document.activeElement !== box.minX) box.minX.value = workspace.min_x;
    if (document.activeElement !== box.maxX) box.maxX.value = workspace.max_x;
    if (document.activeElement !== box.minY) box.minY.value = workspace.min_y;
    if (document.activeElement !== box.maxY) box.maxY.value = workspace.max_y;
    pointsEl.textContent = (next.points || []).map((point, index) => (
      (index + 1) + ".  " + point.x.toFixed(2) + ",  " + point.y.toFixed(2)
    )).join("\n") || "Точок ще немає";
    const selected = routesEl.value;
    routesEl.innerHTML = "";
    (next.routes || []).forEach((route) => {
      const option = document.createElement("option");
      option.value = route.id;
      option.textContent = route.name + " (" + route.points.length + ")";
      routesEl.appendChild(option);
    });
    if (selected) routesEl.value = selected;
    if (!noteEl.dataset.sticky) setNote(next.note || (next.motion_enabled ? "" : "Рух вимкнено, доки команда не перевірена на столі"), !next.motion_enabled && !next.simulated);
    draw();
  }

  function draw() {
    if (!state || !canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#161616";
    ctx.fillRect(0, 0, width, height);
    const points = (state.points || []).slice();
    const pos = state.position_mm || { x: 0, y: 0 };
    const samples = points.concat([pos, { x: 0, y: 0 }]);
    let minX = Math.min(...samples.map((point) => point.x));
    let maxX = Math.max(...samples.map((point) => point.x));
    let minY = Math.min(...samples.map((point) => point.y));
    let maxY = Math.max(...samples.map((point) => point.y));
    const pad = 0.4;
    minX -= pad;
    minY -= pad;
    maxX += pad;
    maxY += pad;
    if (maxX - minX < 1) maxX = minX + 1;
    if (maxY - minY < 1) maxY = minY + 1;
    const scale = Math.min((width - 16) / (maxX - minX), (height - 16) / (maxY - minY));
    const mapX = (x) => 8 + (x - minX) * scale;
    const mapY = (y) => height - 8 - (y - minY) * scale;
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = mapX(point.x);
      const y = mapY(point.y);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    points.forEach((point) => {
      ctx.fillStyle = "#93c5fd";
      ctx.beginPath();
      ctx.arc(mapX(point.x), mapY(point.y), 3, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.fillStyle = "#f8fafc";
    ctx.beginPath();
    ctx.arc(mapX(pos.x), mapY(pos.y), 5, 0, Math.PI * 2);
    ctx.fill();
  }

  async function refresh() {
    if (noteEl.dataset.sticky) return;
    const response = await fetch("/api/stage");
    if (!response.ok) return;
    apply(await response.json());
  }

  panel.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-stage]");
    if (!button || !canMove) return;
    const action = button.dataset.stage;
    noteEl.dataset.sticky = "1";
    if (action === "jog") {
      await send("/api/stage/jog", { method: "POST", body: JSON.stringify({ axis: button.dataset.axis, sign: Number(button.dataset.sign) }) });
    } else if (action === "stop") {
      await send("/api/stage/stop", { method: "POST", body: "{}" });
    } else if (action === "zero") {
      await send("/api/stage/zero", { method: "POST", body: "{}" });
    } else if (action === "add") {
      await send("/api/stage/points", { method: "POST", body: "{}" });
    } else if (action === "pop") {
      await send("/api/stage/points/last", { method: "DELETE" });
    } else if (action === "save") {
      await send("/api/stage/routes", { method: "POST", body: JSON.stringify({ name: nameEl.value }) });
    } else if (action === "load") {
      if (!routesEl.value) return;
      await send("/api/stage/routes/" + routesEl.value + "/load", { method: "POST", body: "{}" });
    } else if (action === "delete") {
      if (!routesEl.value) return;
      await send("/api/stage/routes/" + routesEl.value, { method: "DELETE" });
    } else if (action === "run") {
      await send("/api/stage/run", { method: "POST", body: JSON.stringify({ speed_mm_s: Number(speedEl.value) }) });
    } else if (action === "goryaev") {
      const step = Number(document.getElementById("stageGridStep").value);
      const second = document.getElementById("stageSecondGrid").checked;
      const body = { step_mm: step, width_mm: 3, height_mm: 3 };
      if (second) body.second_offset = { x: Number(document.getElementById("stageOffX").value), y: Number(document.getElementById("stageOffY").value) };
      await send("/api/stage/goryaev", { method: "POST", body: JSON.stringify(body) });
    }
    delete noteEl.dataset.sticky;
  });

  async function saveSettings() {
    await send("/api/stage/settings", {
      method: "PUT",
      body: JSON.stringify({
        speed_mm_s: Number(speedEl.value),
        step_mm: Number(stepEl.value),
        workspace_mm: {
          min_x: Number(box.minX.value),
          max_x: Number(box.maxX.value),
          min_y: Number(box.minY.value),
          max_y: Number(box.maxY.value),
        },
      }),
    });
  }

  [speedEl, stepEl, box.minX, box.maxX, box.minY, box.maxY].forEach((input) => {
    input.addEventListener("change", () => {
      if (canMove) saveSettings();
    });
  });

  if (!canMove) {
    panel.querySelectorAll("button, input, select").forEach((el) => {
      el.disabled = true;
    });
  }

  refresh();
  setInterval(refresh, 700);
})();
