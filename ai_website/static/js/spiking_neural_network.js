(() => {
    const networkCanvas = document.getElementById('snnNetwork');
    const rasterCanvas = document.getElementById('snnRaster');
    const voltageCanvas = document.getElementById('snnVoltage');
    if (!(networkCanvas instanceof HTMLCanvasElement) ||
        !(rasterCanvas instanceof HTMLCanvasElement) ||
        !(voltageCanvas instanceof HTMLCanvasElement)) return;

    const networkContext = networkCanvas.getContext('2d');
    const rasterContext = rasterCanvas.getContext('2d');
    const voltageContext = voltageCanvas.getContext('2d');
    if (!networkContext || !rasterContext || !voltageContext) return;

    const neuronCount = 16;
    const restPotential = 0;
    const refractorySteps = 3;
    const historyLength = 240;
    const timeReadout = document.getElementById('simTime');
    const spikeReadout = document.getElementById('spikeCount');
    const rateReadout = document.getElementById('rateReadout');
    const runStatus = document.getElementById('runStatus');
    const runIndicator = document.getElementById('runIndicator');
    const runButton = document.getElementById('toggleRun');
    const stepButton = document.getElementById('stepSimulation');
    const resetButton = document.getElementById('resetSimulation');
    const eventList = document.getElementById('snnEvents');
    const currentInput = document.getElementById('inputCurrent');
    const thresholdInput = document.getElementById('threshold');
    const tauInput = document.getElementById('membraneTau');
    const inhibitionInput = document.getElementById('inhibition');
    const patternInput = document.getElementById('stimulusPattern');
    let neurons = [];
    let connections = [];
    let pendingEvents = [];
    let raster = [];
    let voltageHistory = [];
    let tick = 0;
    let totalSpikes = 0;
    let running = false;
    let timer = null;

    function makeNetwork() {
        neurons = Array.from({ length: neuronCount }, (_, id) => ({
            id,
            layer: id < 4 ? 0 : id < 12 ? 1 : 2,
            x: id < 4 ? 105 : id < 12 ? 440 : 790,
            y: id < 4
                ? 75 + (id + 0.5) * 45
                : id < 12
                    ? 40 + (id - 3.5) * 31
                    : 70 + (id - 11.5) * 47,
            voltage: restPotential,
            refractory: 0,
            synapticCurrent: 0,
            lastSpike: -Infinity
        }));

        connections = [];
        for (let source = 0; source < neuronCount; source += 1) {
            const targets = [];
            if (source < 4) {
                targets.push(4 + source, 4 + ((source + 1) % 8), 4 + ((source + 4) % 8));
            } else if (source < 12) {
                targets.push(4 + ((source - 3) % 8), 4 + ((source - 2) % 8), 12 + ((source - 4) % 4));
            } else {
                targets.push(4 + ((source - 12) % 8));
            }
            [...new Set(targets)].forEach((target, index) => {
                const inhibitory = source >= 4 && source < 12 && (source + target + index) % 5 === 0;
                const weight = 1.0 + ((source * 5 + target * 3 + index) % 5) * 0.16;
                connections.push({ source, target, weight, inhibitory });
            });
        }

        // Give the network a small, stable recurrent feedback loop.
        [[6, 9], [9, 6], [7, 10], [10, 7]].forEach(([source, target], index) => {
            connections.push({ source, target, weight: 1.0 + index * 0.1, inhibitory: index === 2 });
        });
    }

    function readParameters() {
        return {
            current: Number(currentInput.value),
            threshold: Number(thresholdInput.value),
            tau: Number(tauInput.value),
            inhibition: Number(inhibitionInput.value),
            pattern: patternInput.value
        };
    }

    function externalCurrent(neuronId, parameters) {
        if (neuronId >= 4) return 0;
        const phase = tick % 120;
        let patternScale = 1;
        if (parameters.pattern === 'burst') patternScale = phase < 22 ? 1.45 : 0;
        if (parameters.pattern === 'ramp') patternScale = 0.25 + 0.9 * (phase / 119);
        return parameters.current * patternScale * (0.92 + neuronId * 0.055);
    }

    function recordEvent(neuronId) {
        if (!eventList) return;
        const empty = eventList.querySelector('.snn-empty');
        if (empty) empty.remove();
        const item = document.createElement('li');
        const time = document.createElement('time');
        time.textContent = `${tick} ms`;
        item.append(time, document.createTextNode(`Neuron ${neuronId + 1} fired`));
        eventList.prepend(item);
        while (eventList.children.length > 8) eventList.lastElementChild.remove();
    }

    function simulateStep() {
        const parameters = readParameters();
        const incoming = pendingEvents.filter((event) => event.deliveryTick <= tick);
        pendingEvents = pendingEvents.filter((event) => event.deliveryTick > tick);
        incoming.forEach((event) => {
            const neuron = neurons[event.target];
            neuron.synapticCurrent += event.inhibitory
                ? -parameters.inhibition * event.weight
                : event.weight;
        });

        const newlyFired = [];
        neurons.forEach((neuron) => {
            if (neuron.refractory > 0) {
                neuron.refractory -= 1;
                neuron.voltage = restPotential;
            } else {
                const input = externalCurrent(neuron.id, parameters);
                const voltageDelta = (-(neuron.voltage - restPotential) + input + neuron.synapticCurrent) / parameters.tau;
                neuron.voltage += voltageDelta;
                if (neuron.voltage >= parameters.threshold) {
                    neuron.voltage = restPotential;
                    neuron.refractory = refractorySteps;
                    neuron.lastSpike = tick;
                    newlyFired.push(neuron.id);
                }
            }
            neuron.synapticCurrent *= 0.88;
        });

        newlyFired.forEach((source) => {
            totalSpikes += 1;
            recordEvent(source);
            connections.filter((connection) => connection.source === source).forEach((connection) => {
                pendingEvents.push({
                    source,
                    target: connection.target,
                    weight: connection.weight,
                    inhibitory: connection.inhibitory,
                    deliveryTick: tick + 2,
                    sentTick: tick
                });
            });
        });

        raster.push({ tick, neurons: newlyFired });
        if (raster.length > historyLength) raster.shift();
        voltageHistory.push({ tick, voltage: neurons[7].voltage });
        if (voltageHistory.length > historyLength) voltageHistory.shift();
        tick += 1;
        updateReadouts();
        draw();
    }

    function updateReadouts() {
        timeReadout.textContent = String(tick);
        spikeReadout.textContent = String(totalSpikes);
        const rateWindow = raster.slice(-100);
        const recentSpikes = rateWindow.reduce((total, entry) => total + entry.neurons.length, 0);
        rateReadout.textContent = `${(recentSpikes / (Math.max(rateWindow.length, 1) * neuronCount) * 1000).toFixed(1)} Hz`;
    }

    function prepareCanvas(canvas, context, logicalWidth, logicalHeight) {
        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        const displayWidth = canvas.clientWidth || logicalWidth;
        const displayHeight = displayWidth * logicalHeight / logicalWidth;
        if (canvas.width !== Math.round(displayWidth * pixelRatio) || canvas.height !== Math.round(displayHeight * pixelRatio)) {
            canvas.width = Math.round(displayWidth * pixelRatio);
            canvas.height = Math.round(displayHeight * pixelRatio);
        }
        context.setTransform(canvas.width / logicalWidth, 0, 0, canvas.height / logicalHeight, 0, 0);
        context.clearRect(0, 0, logicalWidth, logicalHeight);
        return displayWidth / logicalWidth;
    }

    function drawNetwork() {
        prepareCanvas(networkCanvas, networkContext, 900, 330);
        connections.forEach((connection) => {
            const source = neurons[connection.source];
            const target = neurons[connection.target];
            networkContext.beginPath();
            networkContext.moveTo(source.x, source.y);
            networkContext.lineTo(target.x, target.y);
            networkContext.strokeStyle = connection.inhibitory ? 'rgba(218, 113, 109, 0.34)' : 'rgba(50, 139, 143, 0.24)';
            networkContext.lineWidth = 0.8 + connection.weight;
            networkContext.stroke();
        });

        pendingEvents.forEach((event) => {
            const source = neurons[event.source];
            const target = neurons[event.target];
            const progress = 1 - (event.deliveryTick - tick) / 2;
            const x = source.x + (target.x - source.x) * progress;
            const y = source.y + (target.y - source.y) * progress;
            networkContext.beginPath();
            networkContext.arc(x, y, 3.5, 0, Math.PI * 2);
            networkContext.fillStyle = event.inhibitory ? '#e17c77' : '#1db6a3';
            networkContext.fill();
        });

        neurons.forEach((neuron) => {
            const firedRecently = neuron.lastSpike > -Infinity && tick - neuron.lastSpike <= 8;
            const radius = neuron.layer === 1 ? 9 : 11;
            networkContext.beginPath();
            networkContext.arc(neuron.x, neuron.y, radius + (firedRecently ? 8 : 0), 0, Math.PI * 2);
            networkContext.fillStyle = firedRecently ? 'rgba(22, 181, 163, 0.16)' : 'rgba(0,0,0,0)';
            networkContext.fill();
            networkContext.beginPath();
            networkContext.arc(neuron.x, neuron.y, radius, 0, Math.PI * 2);
            networkContext.fillStyle = firedRecently ? '#1db6a3' : neuron.layer === 0 ? '#e8a451' : neuron.layer === 2 ? '#8386dc' : '#f8fbfc';
            networkContext.fill();
            networkContext.lineWidth = 1.5;
            networkContext.strokeStyle = firedRecently ? '#087f78' : neuron.layer === 0 ? '#d79138' : neuron.layer === 2 ? '#777aca' : '#78909d';
            networkContext.stroke();
            if (neuron.layer !== 1) {
                networkContext.font = '10px Inter, sans-serif';
                networkContext.textAlign = 'center';
                networkContext.fillStyle = '#596c78';
                networkContext.fillText(neuron.layer === 0 ? `IN ${neuron.id + 1}` : `OUT ${neuron.id - 11}`, neuron.x, neuron.y + 24);
            }
        });
    }

    function drawRaster() {
        prepareCanvas(rasterCanvas, rasterContext, 900, 220);
        const left = 48;
        const right = 12;
        const top = 12;
        const bottom = 28;
        const plotWidth = 900 - left - right;
        const plotHeight = 220 - top - bottom;
        rasterContext.strokeStyle = '#e7edf1';
        rasterContext.fillStyle = '#6b7b89';
        rasterContext.font = '10px Inter, sans-serif';
        for (let neuron = 0; neuron < neuronCount; neuron += 1) {
            const y = top + neuron * plotHeight / (neuronCount - 1);
            rasterContext.beginPath();
            rasterContext.moveTo(left, y);
            rasterContext.lineTo(900 - right, y);
            rasterContext.stroke();
            if (neuron % 2 === 0) {
                rasterContext.textAlign = 'right';
                rasterContext.fillText(`N${neuron + 1}`, left - 9, y + 3);
            }
        }
        raster.forEach((entry) => entry.neurons.forEach((neuron) => {
            const x = left + (entry.tick - Math.max(0, tick - historyLength)) / historyLength * plotWidth;
            const y = top + neuron * plotHeight / (neuronCount - 1);
            rasterContext.beginPath();
            rasterContext.moveTo(x, y - 3.5);
            rasterContext.lineTo(x, y + 3.5);
            rasterContext.strokeStyle = neuron < 4 ? '#e8a451' : neuron >= 12 ? '#8386dc' : '#0f9f97';
            rasterContext.lineWidth = 2;
            rasterContext.stroke();
        }));
        rasterContext.textAlign = 'left';
        rasterContext.fillStyle = '#71808d';
        rasterContext.fillText(`${Math.max(0, tick - historyLength)} ms`, left, 211);
        rasterContext.textAlign = 'right';
        rasterContext.fillText(`${tick} ms`, 900 - right, 211);
    }

    function drawVoltage() {
        prepareCanvas(voltageCanvas, voltageContext, 900, 220);
        const left = 48;
        const right = 12;
        const top = 13;
        const bottom = 29;
        const plotWidth = 900 - left - right;
        const plotHeight = 220 - top - bottom;
        const threshold = readParameters().threshold;
        const maxVoltage = Math.max(1.6, threshold * 1.2);
        const yFor = (voltage) => top + plotHeight - Math.max(0, voltage) / maxVoltage * plotHeight;

        [0, threshold, maxVoltage].forEach((value) => {
            const y = yFor(value);
            voltageContext.beginPath();
            voltageContext.setLineDash(value === threshold ? [5, 4] : []);
            voltageContext.moveTo(left, y);
            voltageContext.lineTo(900 - right, y);
            voltageContext.strokeStyle = value === threshold ? '#df7770' : '#e7edf1';
            voltageContext.lineWidth = value === threshold ? 1.5 : 1;
            voltageContext.stroke();
            voltageContext.setLineDash([]);
            voltageContext.fillStyle = value === threshold ? '#c66d68' : '#71808d';
            voltageContext.font = '10px Inter, sans-serif';
            voltageContext.textAlign = 'right';
            voltageContext.fillText(value === threshold ? 'threshold' : value.toFixed(1), left - 8, y + 3);
        });

        if (voltageHistory.length > 1) {
            voltageContext.beginPath();
            voltageHistory.forEach((entry, index) => {
                const x = left + index / (historyLength - 1) * plotWidth;
                const y = yFor(entry.voltage);
                if (index === 0) voltageContext.moveTo(x, y);
                else voltageContext.lineTo(x, y);
            });
            voltageContext.strokeStyle = '#4384ed';
            voltageContext.lineWidth = 2;
            voltageContext.stroke();
        }

        voltageContext.textAlign = 'left';
        voltageContext.fillStyle = '#71808d';
        voltageContext.fillText(`${Math.max(0, tick - historyLength)} ms`, left, 211);
        voltageContext.textAlign = 'right';
        voltageContext.fillText(`${tick} ms`, 900 - right, 211);
    }

    function draw() {
        drawNetwork();
        drawRaster();
        drawVoltage();
    }

    function setRunning(nextState) {
        running = nextState;
        if (timer) window.clearInterval(timer);
        timer = null;
        runIndicator.classList.toggle('is-running', running);
        runStatus.textContent = running ? 'Running · 1 ms time steps' : 'Paused · ready to simulate';
        runButton.innerHTML = running
            ? '<i class="fas fa-pause" aria-hidden="true"></i> Pause'
            : '<i class="fas fa-play" aria-hidden="true"></i> Run';
        if (running) timer = window.setInterval(simulateStep, 30);
    }

    function resetSimulation() {
        setRunning(false);
        tick = 0;
        totalSpikes = 0;
        pendingEvents = [];
        raster = [];
        makeNetwork();
        voltageHistory = [{ tick: 0, voltage: restPotential }];
        timeReadout.textContent = '0';
        spikeReadout.textContent = '0';
        rateReadout.textContent = '0.0 Hz';
        eventList.innerHTML = '<li class="snn-empty">Run the network to see firing events.</li>';
        draw();
    }

    runButton.addEventListener('click', () => setRunning(!running));
    stepButton.addEventListener('click', () => {
        setRunning(false);
        for (let index = 0; index < 10; index += 1) simulateStep();
    });
    resetButton.addEventListener('click', resetSimulation);
    [currentInput, thresholdInput, tauInput, inhibitionInput].forEach((input) => {
        input.addEventListener('input', () => {
            document.getElementById('currentValue').textContent = Number(currentInput.value).toFixed(2);
            document.getElementById('thresholdValue').textContent = Number(thresholdInput.value).toFixed(2);
            document.getElementById('tauValue').textContent = `${tauInput.value} ms`;
            document.getElementById('inhibitionValue').textContent = Number(inhibitionInput.value).toFixed(2);
            draw();
        });
    });
    patternInput.addEventListener('change', () => {
        runStatus.textContent = running ? 'Running · 1 ms time steps' : `Paused · ${patternInput.options[patternInput.selectedIndex].text.toLowerCase()} selected`;
    });
    window.addEventListener('resize', draw);
    window.addEventListener('beforeunload', () => {
        if (timer) window.clearInterval(timer);
    }, { once: true });

    resetSimulation();
})();
