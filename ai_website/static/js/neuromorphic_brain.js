(() => {
    const canvas = document.getElementById('brainCanvas');
    if (!(canvas instanceof HTMLCanvasElement)) return;

    const context = canvas.getContext('2d');
    if (!context) return;

    const firingCount = document.getElementById('firingCount');
    const activeCount = document.getElementById('activeCount');
    const systemState = document.getElementById('systemState');
    const responseTitle = document.getElementById('responseTitle');
    const responseDescription = document.getElementById('responseDescription');
    const responseIcon = document.getElementById('responseIcon');
    const eventLog = document.getElementById('eventLog');
    const strengthSlider = document.getElementById('signalStrength');
    const strengthValue = document.getElementById('strengthValue');
    const sequenceButton = document.getElementById('runSequence');
    const resetButton = document.getElementById('resetNetwork');

    const width = 900;
    const height = 560;
    const layers = [
        { id: 'sensor', label: 'SENSORY INPUT', count: 3, x: 92 },
        { id: 'relay', label: 'SPIKE RELAY', count: 5, x: 310 },
        { id: 'association', label: 'ASSOCIATION', count: 6, x: 530 },
        { id: 'response', label: 'MOTOR / OUTPUT', count: 3, x: 770 }
    ];
    const sensorTypes = [
        { id: 'vision', label: 'Light', node: 'sensor-0', output: 'response-0', response: 'Orient to visual input', detail: 'Visual signal reached the attention output.', icon: 'fa-eye' },
        { id: 'sound', label: 'Sound', node: 'sensor-1', output: 'response-1', response: 'Attend to sound', detail: 'Auditory signal reached the alert output.', icon: 'fa-volume-up' },
        { id: 'touch', label: 'Touch', node: 'sensor-2', output: 'response-2', response: 'Trigger touch reflex', detail: 'Tactile signal reached the reflex output.', icon: 'fa-hand' }
    ];
    const nodes = [];
    const nodeById = new Map();
    const edges = [];
    const pulses = [];
    const activeWindows = new Map();
    let totalSpikes = 0;
    let animationFrame = 0;
    let sequenceTimers = [];
    let activityVersion = 0;

    layers.forEach((layer, layerIndex) => {
        for (let index = 0; index < layer.count; index += 1) {
            const node = {
                id: `${layer.id}-${index}`,
                layer: layerIndex,
                x: layer.x,
                y: 130 + ((index + 1) * 300) / (layer.count + 1),
                label: layer.id === 'sensor' ? ['VISION', 'SOUND', 'TOUCH'][index] :
                    layer.id === 'response' ? ['FOCUS', 'ALERT', 'REFLEX'][index] : ''
            };
            nodes.push(node);
            nodeById.set(node.id, node);
        }
    });

    for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex += 1) {
        const sourceLayer = layers[layerIndex];
        const targetLayer = layers[layerIndex + 1];
        for (let sourceIndex = 0; sourceIndex < sourceLayer.count; sourceIndex += 1) {
            const targetIndices = layerIndex === 2
                ? [sourceIndex % targetLayer.count, (sourceIndex + 1) % targetLayer.count]
                : [sourceIndex % targetLayer.count, (sourceIndex + 1) % targetLayer.count, (sourceIndex + 2) % targetLayer.count];
            [...new Set(targetIndices)].forEach((targetIndex, connectionIndex) => {
                edges.push({
                    from: `${sourceLayer.id}-${sourceIndex}`,
                    to: `${targetLayer.id}-${targetIndex}`,
                    weight: 0.45 + ((sourceIndex * 7 + targetIndex * 3 + connectionIndex) % 6) / 10
                });
            });
        }
    }

    const synapseCount = document.getElementById('synapseCount');
    if (synapseCount) synapseCount.textContent = String(edges.length);

    function addEvent(message) {
        if (!eventLog) return;
        const empty = eventLog.querySelector('.event-empty');
        if (empty) empty.remove();
        const item = document.createElement('li');
        const time = document.createElement('span');
        time.className = 'event-time';
        time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        item.append(time, document.createTextNode(message));
        eventLog.prepend(item);
        while (eventLog.children.length > 6) eventLog.lastElementChild.remove();
    }

    function setActive(nodeId, start, until) {
        const existing = activeWindows.get(nodeId);
        if (existing) {
            existing.start = Math.min(existing.start, start);
            existing.until = Math.max(existing.until, until);
        } else {
            activeWindows.set(nodeId, { start, until });
        }
    }

    function setResponse(sensor) {
        if (responseTitle) responseTitle.textContent = sensor.response;
        if (responseDescription) responseDescription.textContent = sensor.detail;
        if (responseIcon) responseIcon.innerHTML = `<i class="fas ${sensor.icon}" aria-hidden="true"></i>`;
    }

    function stimulate(sensorId) {
        const sensor = sensorTypes.find((item) => item.id === sensorId);
        if (!sensor) return;

        const now = performance.now();
        const version = ++activityVersion;
        const intensity = Number(strengthSlider?.value || 75) / 100;
        const interval = 470;
        let frontier = new Set([sensor.node]);
        const plannedEdges = [];

        layers.slice(0, -1).forEach((layer, stage) => {
            const outgoing = edges.filter((edge) =>
                frontier.has(edge.from) &&
                (stage !== layers.length - 2 || edge.to === sensor.output)
            );
            const nextFrontier = new Set();
            outgoing.forEach((edge, index) => {
                plannedEdges.push({ ...edge, start: now + stage * interval + (index % 4) * 38, intensity });
                nextFrontier.add(edge.to);
            });
            frontier = nextFrontier;
        });

        setActive(sensor.node, now, now + 600);
        plannedEdges.forEach((pulse) => {
            pulse.counted = false;
            pulses.push(pulse);
            setActive(pulse.to, pulse.start + 440, pulse.start + 900);
        });
        totalSpikes += 1;
        if (firingCount) firingCount.textContent = String(totalSpikes);
        if (systemState) {
            systemState.textContent = `Processing · ${sensor.label} input`;
            systemState.classList.add('is-active');
        }
        setResponse(sensor);
        addEvent(`${sensor.label} input · ${plannedEdges.length + 1} spikes scheduled`);

        const completion = now + (layers.length - 1) * interval + 850;
        window.setTimeout(() => {
            if (systemState && version === activityVersion) {
                systemState.textContent = 'Response complete · ready';
                systemState.classList.remove('is-active');
            }
        }, completion - now);
    }

    function runSequence() {
        sequenceTimers.forEach(window.clearTimeout);
        sequenceTimers = [];
        const sequence = ['vision', 'sound', 'touch'];
        sequence.forEach((sensor, index) => {
            sequenceTimers.push(window.setTimeout(() => {
                stimulate(sensor);
                if (index === sequence.length - 1) sequenceTimers = [];
            }, index * 1050));
        });
        addEvent('Three-sense input sequence started');
    }

    function reset() {
        activityVersion += 1;
        sequenceTimers.forEach(window.clearTimeout);
        sequenceTimers = [];
        pulses.length = 0;
        activeWindows.clear();
        totalSpikes = 0;
        if (firingCount) firingCount.textContent = '0';
        if (activeCount) activeCount.textContent = '0';
        if (systemState) {
            systemState.textContent = 'Idle · awaiting input';
            systemState.classList.remove('is-active');
        }
        if (responseTitle) responseTitle.textContent = 'Ready for a signal';
        if (responseDescription) responseDescription.textContent = 'Choose Light, Sound, or Touch to start a spike train.';
        if (responseIcon) responseIcon.innerHTML = '<i class="fas fa-hourglass-half" aria-hidden="true"></i>';
        if (eventLog) eventLog.innerHTML = '<li class="event-empty">Waiting for the first input…</li>';
        draw(performance.now());
    }

    function draw(now) {
        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        const displayWidth = canvas.clientWidth || width;
        const displayHeight = displayWidth * height / width;
        if (canvas.width !== Math.round(displayWidth * pixelRatio) || canvas.height !== Math.round(displayHeight * pixelRatio)) {
            canvas.width = Math.round(displayWidth * pixelRatio);
            canvas.height = Math.round(displayHeight * pixelRatio);
        }
        context.setTransform(canvas.width / width, 0, 0, canvas.height / height, 0, 0);
        context.clearRect(0, 0, width, height);

        const scale = displayWidth / width;
        const activeNodes = new Set();
        activeWindows.forEach((window, nodeId) => {
            if (window.start <= now && window.until > now) activeNodes.add(nodeId);
            else if (window.until <= now) activeWindows.delete(nodeId);
        });

        edges.forEach((edge) => {
            const from = nodeById.get(edge.from);
            const to = nodeById.get(edge.to);
            context.beginPath();
            context.moveTo(from.x, from.y);
            context.lineTo(to.x, to.y);
            context.strokeStyle = 'rgba(105, 147, 165, 0.22)';
            context.lineWidth = 0.7 + edge.weight * 0.55;
            context.stroke();
        });

        pulses.forEach((pulse) => {
            if (!pulse.counted && now >= pulse.start) {
                pulse.counted = true;
                totalSpikes += 1;
                if (firingCount) firingCount.textContent = String(totalSpikes);
            }
            const progress = (now - pulse.start) / 440;
            if (progress < 0 || progress > 1) return;
            const from = nodeById.get(pulse.from);
            const to = nodeById.get(pulse.to);
            const x = from.x + (to.x - from.x) * progress;
            const y = from.y + (to.y - from.y) * progress;
            const radius = (3.2 + pulse.intensity * 2.2) * (0.8 + 0.2 * Math.sin(progress * Math.PI));
            context.beginPath();
            context.arc(x, y, radius * 2.3, 0, Math.PI * 2);
            context.fillStyle = `rgba(84, 227, 208, ${0.1 * pulse.intensity})`;
            context.fill();
            context.beginPath();
            context.arc(x, y, radius, 0, Math.PI * 2);
            context.fillStyle = `rgba(155, 255, 235, ${0.45 + 0.55 * pulse.intensity})`;
            context.fill();
        });

        nodes.forEach((node) => {
            const isActive = activeNodes.has(node.id);
            const isSensor = node.layer === 0;
            const isOutput = node.layer === layers.length - 1;
            const radius = isSensor || isOutput ? 13 : 10;
            const color = isActive ? '#54e3d0' : isSensor ? '#ffb45e' : isOutput ? '#bf9aff' : '#8299ad';

            context.beginPath();
            context.arc(node.x, node.y, radius + (isActive ? 12 : 0), 0, Math.PI * 2);
            context.fillStyle = isActive ? `rgba(84, 227, 208, ${0.12 + Math.sin(now / 90) * 0.035})` : 'rgba(0, 0, 0, 0)';
            context.fill();
            context.beginPath();
            context.arc(node.x, node.y, radius, 0, Math.PI * 2);
            context.fillStyle = isActive ? '#54e3d0' : '#152a37';
            context.fill();
            context.lineWidth = isActive ? 2.5 : 1.5;
            context.strokeStyle = color;
            context.stroke();
            if (isActive) {
                context.beginPath();
                context.arc(node.x, node.y, 3.3, 0, Math.PI * 2);
                context.fillStyle = '#f2fffb';
                context.fill();
            }
            if (node.label) {
                context.textAlign = 'center';
                context.font = `700 ${9 / scale}px Inter, sans-serif`;
                context.fillStyle = isActive ? '#c9fff3' : '#8ea5b1';
                context.fillText(node.label, node.x, node.y + 30);
            }
        });

        for (let index = pulses.length - 1; index >= 0; index -= 1) {
            if (now - pulses[index].start > 440) pulses.splice(index, 1);
        }
        if (activeCount) activeCount.textContent = String(activeNodes.size);
        animationFrame = window.requestAnimationFrame(draw);
    }

    document.querySelectorAll('.sensor-button').forEach((button) => {
        button.addEventListener('click', () => stimulate(button.dataset.sensor));
    });
    document.addEventListener('keydown', (event) => {
        if (event.altKey || event.ctrlKey || event.metaKey ||
            event.target instanceof HTMLInputElement ||
            event.target instanceof HTMLButtonElement ||
            event.target instanceof HTMLTextAreaElement ||
            event.target instanceof HTMLSelectElement) return;
        const sensorByKey = { v: 'vision', a: 'sound', t: 'touch' };
        const sensorId = sensorByKey[event.key.toLowerCase()];
        if (sensorId) stimulate(sensorId);
    });
    sequenceButton?.addEventListener('click', runSequence);
    resetButton?.addEventListener('click', reset);
    strengthSlider?.addEventListener('input', () => {
        if (strengthValue) strengthValue.textContent = `${strengthSlider.value}%`;
    });
    window.addEventListener('beforeunload', () => {
        window.cancelAnimationFrame(animationFrame);
        sequenceTimers.forEach(window.clearTimeout);
    }, { once: true });

    draw(performance.now());
})();
