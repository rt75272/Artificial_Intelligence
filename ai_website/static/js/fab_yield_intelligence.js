const controlsContainer = document.getElementById('fabControls');
const runForecastBtn = document.getElementById('runForecast');

if (controlsContainer && runForecastBtn) {
  const fabIds = (controlsContainer.dataset.featureIds || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  const fabApiUrl = runForecastBtn.dataset.apiUrl;
  const fabResultsEl = {
    predYield: document.getElementById('predYield'),
    riskLevel: document.getElementById('riskLevel'),
    confidenceWindow: document.getElementById('confidenceWindow'),
    topDrivers: document.getElementById('topDrivers'),
    recommendations: document.getElementById('recommendations')
  };

  const hasAllResultNodes = Object.values(fabResultsEl).every(Boolean);

  function updateFabRangeLabels() {
    fabIds.forEach((id) => {
      const input = document.getElementById(id);
      const output = document.getElementById(`${id}_value`);
      if (input && output) output.textContent = input.value;
    });
  }

  function collectFabPayload() {
    const payload = {};
    fabIds.forEach((id) => {
      const input = document.getElementById(id);
      if (input) payload[id] = parseFloat(input.value);
    });
    return payload;
  }

  async function runFabForecast() {
    if (!fabApiUrl || !hasAllResultNodes) return;
    try {
      const response = await fetch(fabApiUrl, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(collectFabPayload())
      });
      let data = null;
      try {
        data = await response.json();
      } catch (parseError) {
        if (!response.ok) {
          const fallbackMessage = await response.text();
          throw new Error(fallbackMessage || 'Failed to forecast yield');
        }
        throw parseError;
      }
      if (!response.ok) throw new Error(data?.error || 'Failed to forecast yield');

      fabResultsEl.predYield.textContent = `${data.predicted_yield}%`;
      fabResultsEl.riskLevel.textContent = data.risk_level;
      fabResultsEl.riskLevel.style.color =
        data.risk_level === 'Low' ? '#10b981' : data.risk_level === 'Moderate' ? '#f59e0b' : '#ef4444';
      fabResultsEl.confidenceWindow.textContent = `${data.confidence_window.low}% to ${data.confidence_window.high}%`;

      fabResultsEl.topDrivers.innerHTML = '';
      (data.top_drivers || []).forEach((driver) => {
        const item = document.createElement('li');
        item.textContent = `${driver.name}: ${driver.importance_pct}% model influence`;
        fabResultsEl.topDrivers.appendChild(item);
      });

      fabResultsEl.recommendations.innerHTML = '';
      (data.recommendations || []).forEach((rec) => {
        const item = document.createElement('li');
        item.textContent = rec;
        fabResultsEl.recommendations.appendChild(item);
      });
    } catch (error) {
      fabResultsEl.predYield.textContent = 'Error';
      fabResultsEl.riskLevel.textContent = error.message;
      fabResultsEl.confidenceWindow.textContent = '--';
      fabResultsEl.topDrivers.innerHTML = '';
      fabResultsEl.recommendations.innerHTML = '';
    }
  }

  fabIds.forEach((id) => {
    const input = document.getElementById(id);
    if (input) input.addEventListener('input', updateFabRangeLabels);
  });
  runForecastBtn.addEventListener('click', runFabForecast);
  updateFabRangeLabels();
  runFabForecast();
}
