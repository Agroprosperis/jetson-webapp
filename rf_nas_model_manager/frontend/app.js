const state = {
  apiKey: '',
  workspace: null,
  projects: [],
  selectedProject: null,
  tensorRtAvailable: false,
  activeJobId: null,
  pollTimer: null,
};

const byId = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function setMessage(element, text, kind = '') {
  element.textContent = text || '';
  element.className = `message${kind ? ` ${kind}` : ''}`;
}

async function apiRequest(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json') ? await response.json() : null;
  if (!response.ok) {
    throw new Error((data && data.error) || `Request failed (${response.status}).`);
  }
  return data;
}

function strictJson(method, payload) {
  return {
    method,
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  };
}

async function connect(event) {
  event.preventDefault();
  const apiKey = byId('apiKey').value;
  if (!apiKey) return;

  const button = byId('connectButton');
  button.disabled = true;
  setMessage(byId('connectMessage'), 'Loading workspace and projects...');
  try {
    const data = await apiRequest('/api/roboflow/catalog', strictJson('POST', {api_key: apiKey}));
    state.apiKey = apiKey;
    state.workspace = data.workspace;
    state.projects = data.projects;
    renderProjects();
    byId('browsePanel').classList.remove('hidden');
    byId('disconnectButton').classList.remove('hidden');
    setMessage(byId('connectMessage'), `Connected to ${data.workspace.name}.`, 'success');
  } catch (error) {
    state.apiKey = '';
    setMessage(byId('connectMessage'), error.message, 'error');
  } finally {
    button.disabled = false;
  }
}

function disconnect() {
  state.apiKey = '';
  state.workspace = null;
  state.projects = [];
  state.selectedProject = null;
  byId('apiKey').value = '';
  byId('browsePanel').classList.add('hidden');
  byId('remoteModelsPanel').classList.add('hidden');
  byId('disconnectButton').classList.add('hidden');
  setMessage(byId('connectMessage'), '');
}

function renderProjects() {
  byId('workspaceLabel').textContent = `${state.workspace.name} · ${state.workspace.id}`;
  const container = byId('projects');
  if (!state.projects.length) {
    container.innerHTML = '<div class="empty-state">No projects found in this workspace.</div>';
    return;
  }
  container.innerHTML = state.projects.map((project) => `
    <button class="project-card" type="button" data-project-id="${escapeHtml(project.id)}">
      <strong>${escapeHtml(project.name)}</strong>
      <span>${escapeHtml(project.type)} · ${escapeHtml(project.id)}</span>
    </button>
  `).join('');
  container.querySelectorAll('.project-card').forEach((button) => {
    button.addEventListener('click', () => loadRemoteModels(button.dataset.projectId, button));
  });
}

async function loadRemoteModels(projectId, selectedButton) {
  state.selectedProject = projectId;
  byId('projects').querySelectorAll('.project-card').forEach((button) => {
    button.classList.toggle('selected', button === selectedButton);
  });
  byId('remoteModelsPanel').classList.remove('hidden');
  byId('selectedProject').textContent = projectId;
  byId('remoteModels').innerHTML = '';
  setMessage(byId('remoteModelsMessage'), 'Loading trained models...');
  try {
    const data = await apiRequest('/api/roboflow/models', strictJson('POST', {
      api_key: state.apiKey,
      project_id: projectId,
    }));
    renderRemoteModels(data.models);
    setMessage(
      byId('remoteModelsMessage'),
      data.models.length ? '' : 'No trained model versions were found in this project.',
    );
  } catch (error) {
    setMessage(byId('remoteModelsMessage'), error.message, 'error');
  }
}

function renderRemoteModels(models) {
  const container = byId('remoteModels');
  const nasGroups = new Map();
  const ordinaryModels = [];

  models.forEach((model) => {
    if (!model.nas_group) {
      ordinaryModels.push(model);
      return;
    }
    if (!nasGroups.has(model.nas_group)) {
      nasGroups.set(model.nas_group, {version: model.version, models: []});
    }
    nasGroups.get(model.nas_group).models.push(model);
  });

  const compareLatency = (left, right) => {
    const leftHasLatency = Number.isFinite(left.latency);
    const rightHasLatency = Number.isFinite(right.latency);
    if (leftHasLatency && rightHasLatency) return left.latency - right.latency;
    if (leftHasLatency) return -1;
    if (rightHasLatency) return 1;
    return 0;
  };
  const compareF1 = (left, right) => {
    const leftHasF1 = Number.isFinite(left.f1);
    const rightHasF1 = Number.isFinite(right.f1);
    if (leftHasF1 && rightHasF1) return right.f1 - left.f1;
    if (leftHasF1) return -1;
    if (rightHasF1) return 1;
    return 0;
  };
  const sortModels = (groupModels, mode) => [...groupModels].sort((left, right) => {
    if (mode === 'latency-f1') {
      return compareLatency(left, right) || compareF1(left, right) || left.id.localeCompare(right.id);
    }
    if (mode === 'id') return left.id.localeCompare(right.id);
    return compareF1(left, right) || compareLatency(left, right) || left.id.localeCompare(right.id);
  });
  const formatF1 = (value) => Number.isFinite(value) ? String((value * 100).toFixed(1)) + '%' : '—';
  const formatLatency = (value) => Number.isFinite(value) ? value.toFixed(2) + ' ms' : '—';
  const downloadButton = (model) => `
    <button class="model-download-button import-button" type="button"
      data-model-id="${escapeHtml(model.id)}" title="Download and process ${escapeHtml(model.id)}"
      aria-label="Download and process ${escapeHtml(model.id)}">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 3v12m0 0 5-5m-5 5-5-5M5 20h14"/>
      </svg>
    </button>
  `;
  const nasRows = (groupModels, mode = 'f1-latency') => sortModels(groupModels, mode).map((model) => `
    <tr>
      <td class="mono">${escapeHtml(model.id)}</td>
      <td>${escapeHtml(formatF1(model.f1))}</td>
      <td>${escapeHtml(formatLatency(model.latency))}</td>
      <td class="model-action-cell">${downloadButton(model)}</td>
    </tr>
  `).join('');
  const ordinaryRows = ordinaryModels.map((model) => `
    <tr>
      <td class="mono">${escapeHtml(model.id)}</td>
      <td>${escapeHtml(model.version || '—')}</td>
      <td class="model-action-cell">${downloadButton(model)}</td>
    </tr>
  `).join('');

  const nasTables = [...nasGroups.entries()].map(([groupId, group]) => {
    const created = new Date(group.models[0].created * 1000).toLocaleString();
    return `
      <details class="nas-model-group">
        <summary>
          <span>
            <strong>Version ${escapeHtml(group.version)} NAS Results</strong>
            <small>${group.models.length} candidate models · ${escapeHtml(created)}</small>
          </span>
          <span class="nas-group-toggle" aria-hidden="true"></span>
        </summary>
        <div class="nas-table-toolbar">
          <label>
            <span>Sort by</span>
            <select class="nas-sort-select" data-nas-group="${escapeHtml(groupId)}">
              <option value="f1-latency">F1 highest, then latency</option>
              <option value="latency-f1">Latency lowest, then F1</option>
              <option value="id">Model ID</option>
            </select>
          </label>
        </div>
        <div class="remote-model-table-wrap">
          <table class="remote-model-table">
            <thead>
              <tr><th>Model ID</th><th>F1</th><th>Latency</th><th><span class="sr-only">Action</span></th></tr>
            </thead>
            <tbody>${nasRows(group.models)}</tbody>
          </table>
        </div>
      </details>
    `;
  }).join('');

  container.innerHTML = nasTables + (ordinaryRows ? `
    <h3 class="other-models-heading">Other trained models</h3>
    <div class="remote-model-table-wrap">
      <table class="remote-model-table">
        <thead>
          <tr><th>Model ID</th><th>Version</th><th><span class="sr-only">Action</span></th></tr>
        </thead>
        <tbody>${ordinaryRows}</tbody>
      </table>
    </div>
  ` : '');

  const bindImportButtons = (root) => root.querySelectorAll('.import-button').forEach((button) => {
    button.addEventListener('click', () => importModel(button.dataset.modelId, button));
  });
  bindImportButtons(container);
  container.querySelectorAll('.nas-sort-select').forEach((select) => {
    select.addEventListener('change', () => {
      const tbody = select.closest('.nas-model-group').querySelector('tbody');
      tbody.innerHTML = nasRows(nasGroups.get(select.dataset.nasGroup).models, select.value);
      bindImportButtons(tbody);
    });
  });
}

async function importModel(modelId, button) {
  button.disabled = true;
  try {
    const data = await apiRequest('/api/models/import', strictJson('POST', {
      api_key: state.apiKey,
      model_id: modelId,
    }));
    openJob(data.job_id, `Importing ${modelId}`);
  } catch (error) {
    setMessage(byId('remoteModelsMessage'), error.message, 'error');
    button.disabled = false;
  }
}

async function loadCatalog() {
  setMessage(byId('catalogMessage'), 'Loading prepared models...');
  try {
    const data = await apiRequest('/api/models');
    state.tensorRtAvailable = Boolean(data.tensorrt && data.tensorrt.available);
    renderTensorRt(data.tensorrt || {});
    renderCatalog(data.models || []);
    setMessage(byId('catalogMessage'), '');
  } catch (error) {
    setMessage(byId('catalogMessage'), error.message, 'error');
  }
}

function renderTensorRt(tensorrt) {
  const badge = byId('trtStatus');
  if (tensorrt.available) {
    badge.textContent = `TensorRT ${tensorrt.version}`;
    badge.className = 'status-pill success';
  } else {
    badge.textContent = 'TensorRT unavailable';
    badge.className = 'status-pill neutral';
  }
}

function renderCatalog(models) {
  byId('emptyCatalog').classList.toggle('hidden', models.length > 0);
  byId('catalog').classList.toggle('hidden', models.length === 0);
  byId('catalogBody').innerHTML = models.map((model) => {
    const preprocessing = model.preprocessing || {};
    const engine = model.engine || null;
    const engineLabel = engine
      ? `<strong>${escapeHtml(engine.name)}</strong><div class="meta">TensorRT ${escapeHtml(engine.tensorrt_version || 'unknown')}</div>`
      : '<span class="muted">Not compiled</span>';
    return `
      <tr>
        <td><strong>${escapeHtml(model.name)}</strong><div class="meta mono">${escapeHtml(model.model_id)}</div></td>
        <td>${escapeHtml(model.inference_width)} × ${escapeHtml(model.inference_height)}</td>
        <td>
          ${escapeHtml(preprocessing.color_mode || '—')} / ${escapeHtml(preprocessing.resize_mode || '—')}
          <div class="meta">scale ${escapeHtml(preprocessing.scaling_factor ?? '—')}</div>
        </td>
        <td>${engineLabel}</td>
        <td>
          <div class="actions">
            <button class="button small primary compile-button" type="button" data-name="${escapeHtml(model.name)}" ${state.tensorRtAvailable ? '' : 'disabled'}>
              ${model.compiled ? 'Recompile' : 'Compile'}
            </button>
            <a class="button small ghost" href="/api/models/${encodeURIComponent(model.name)}/package">Download ZIP</a>
            <button class="button small danger delete-button" type="button" data-name="${escapeHtml(model.name)}">Delete</button>
          </div>
        </td>
      </tr>
    `;
  }).join('');
  byId('catalogBody').querySelectorAll('.compile-button').forEach((button) => {
    button.addEventListener('click', () => compileModel(button.dataset.name, button));
  });
  byId('catalogBody').querySelectorAll('.delete-button').forEach((button) => {
    button.addEventListener('click', () => deleteModel(button.dataset.name, button));
  });
}

async function compileModel(name, button) {
  button.disabled = true;
  try {
    const data = await apiRequest(`/api/models/${encodeURIComponent(name)}/compile`, {method: 'POST'});
    openJob(data.job_id, `Compiling ${name}`);
  } catch (error) {
    setMessage(byId('catalogMessage'), error.message, 'error');
    button.disabled = false;
  }
}

async function deleteModel(name, button) {
  if (!window.confirm(`Delete ${name} and all of its prepared artifacts?`)) return;
  button.disabled = true;
  try {
    await apiRequest(`/api/models/${encodeURIComponent(name)}`, {method: 'DELETE'});
    await loadCatalog();
  } catch (error) {
    setMessage(byId('catalogMessage'), error.message, 'error');
    button.disabled = false;
  }
}

function openJob(jobId, title) {
  state.activeJobId = jobId;
  byId('jobTitle').textContent = title;
  byId('jobStatus').textContent = 'Queued';
  byId('jobLogs').textContent = '';
  byId('jobOverlay').classList.remove('hidden');
  clearInterval(state.pollTimer);
  pollJob();
  state.pollTimer = window.setInterval(pollJob, 2000);
}

async function pollJob() {
  if (!state.activeJobId) return;
  try {
    const job = await apiRequest(`/api/jobs/${encodeURIComponent(state.activeJobId)}`);
    byId('jobStatus').textContent = job.status;
    byId('jobLogs').textContent = (job.logs || []).join('\n');
    byId('jobLogs').scrollTop = byId('jobLogs').scrollHeight;
    if (job.status === 'done' || job.status === 'error') {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      document.querySelectorAll(".import-button").forEach((button) => { button.disabled = false; });
      if (job.status === 'done') closeJob();
      await loadCatalog();
    }
  } catch (error) {
    byId('jobStatus').textContent = error.message;
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function closeJob() {
  byId('jobOverlay').classList.add('hidden');
}

byId('connectForm').addEventListener('submit', connect);
byId('disconnectButton').addEventListener('click', disconnect);
byId('refreshButton').addEventListener('click', loadCatalog);
byId('closeJobButton').addEventListener('click', closeJob);
loadCatalog();
