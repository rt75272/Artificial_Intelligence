const fabIds = [
  'etch_uniformity',
  'overlay_error',
  'particle_count',
  'chamber_pressure',
  'temperature_delta',
  'tool_age_days',
  'vibration_index'
];

const fabResultsEl = {
  predYield: document.getElementById('predYield'),
  riskLevel: document.getElementById('riskLevel'),
  confidenceWindow: document.getElementById('confidenceWindow'),
  topDrivers: document.getElementById('topDrivers'),
  recommendations: document.getElementById('recommendations')
};

function updateFabRangeLabels() {
  fabIds.forEach((id) => {
    const input = document.getElementById(id);
    const output = document.getElementById(`${id}_value`);
    output.textContent = input.value;
  });
}

function collectFabPayload() {
  const payload = {};
  fabIds.forEach((id) => {
    payload[id] = parseFloat(document.getElementById(id).value);
  });
  return payload;
}

async function runFabForecast() {
  try {
    const response = await fetch('/api/fab-yield', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(collectFabPayload())
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Failed to forecast yield');

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
  document.getElementById(id).addEventListener('input', updateFabRangeLabels);
});
document.getElementById('runForecast').addEventListener('click', runFabForecast);
updateFabRangeLabels();
runFabForecast();
