(() => {
    const root = document.getElementById('scout-game');
    const get = (id) => document.getElementById(id);
    const missions = [
        {
            title: 'The warm welcome',
            rule: 'A signal is a match when its warmth is 6 or higher.',
            match: (warmth) => warmth >= 6,
        },
        {
            title: 'Bright and bold',
            rule: 'A signal is a match when warmth plus brightness is 11 or higher.',
            match: (warmth, brightness) => warmth + brightness >= 11,
        },
        {
            title: 'The cool glow',
            rule: 'A signal is a match when brightness is greater than warmth.',
            match: (warmth, brightness) => brightness > warmth,
        },
    ];
    const trainingSize = 6;
    const testSize = 3;
    let missionIndex, phase, samples, index, weights, playerScore, scoutScore, tested;

    function shuffle(items) {
        const result = items.slice();
        for (let i = result.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [result[i], result[j]] = [result[j], result[i]];
        }
        return result;
    }

    function makeSamples(mission) {
        const all = [];
        for (let warmth = 0; warmth <= 10; warmth++) {
            for (let brightness = 0; brightness <= 10; brightness++) {
                all.push({ warmth, brightness, label: mission.match(warmth, brightness) ? 1 : -1 });
            }
        }
        const matches = shuffle(all.filter((sample) => sample.label === 1));
        const nonMatches = shuffle(all.filter((sample) => sample.label === -1));
        const training = shuffle([...matches.splice(0, 3), ...nonMatches.splice(0, 3)]);
        return { training, testing: shuffle([...matches, ...nonMatches]).slice(0, testSize) };
    }

    function setFeedback(message) {
        get('feedback').textContent = message;
    }

    function updateDashboard() {
        get('mission-count').textContent = `${missionIndex + 1} / ${missions.length}`;
        get('stage-label').textContent = phase === 'training' ? 'Training data' : phase === 'testing' ? 'Test challenge' : 'Complete';
        get('player-score').textContent = `${playerScore} / ${tested}`;
        get('scout-score').textContent = `${scoutScore} / ${tested}`;
    }

    function showSignal() {
        const sample = samples[phase][index];
        get('signal-card').hidden = false;
        get('question').hidden = false;
        get('answer-actions').hidden = false;
        get('continue-button').hidden = true;
        get('train-button').hidden = true;
        get('next-mission-button').hidden = true;
        get('warmth-value').textContent = `${sample.warmth} / 10`;
        get('brightness-value').textContent = `${sample.brightness} / 10`;
        get('warmth-bar').style.width = `${sample.warmth * 10}%`;
        get('brightness-bar').style.width = `${sample.brightness * 10}%`;
        get('round-progress').textContent = `${phase === 'training' ? 'Training signal' : 'Test signal'} ${index + 1} / ${phase === 'training' ? trainingSize : testSize}`;
        get('phase-hint').textContent = phase === 'training' ? 'Label this example' : 'You and the scout predict';
        setFeedback(phase === 'training' ? 'Use the mission rule to label this signal.' : 'Make your prediction before revealing the scout’s answer.');
        updateDashboard();
    }

    function startMission() {
        const mission = missions[missionIndex];
        phase = 'training';
        index = 0;
        weights = null;
        samples = makeSamples(mission);
        get('mission-number').textContent = `Mission 0${missionIndex + 1}`;
        get('mission-title').textContent = mission.title;
        get('mission-rule').textContent = mission.rule;
        get('learning-text').textContent = 'A training example is a signal paired with the right answer. Your labels are checked before the scout learns from them.';
        get('model-status').textContent = 'Not trained yet';
        ['weight-warmth', 'weight-brightness', 'weight-bias'].forEach((id) => { get(id).textContent = '—'; });
        showSignal();
    }

    function predict(sample) {
        return weights[0] * sample.warmth / 10 + weights[1] * sample.brightness / 10 + weights[2] >= 0 ? 1 : -1;
    }

    function train() {
        weights = [0, 0, 0];
        let updates = 0;
        for (let epoch = 0; epoch < 20; epoch++) {
            for (const sample of samples.training) {
                if (predict(sample) !== sample.label) {
                    weights[0] += sample.label * sample.warmth / 10;
                    weights[1] += sample.label * sample.brightness / 10;
                    weights[2] += sample.label;
                    updates++;
                }
            }
        }
        const correct = samples.training.filter((sample) => predict(sample) === sample.label).length;
        get('model-status').textContent = `${correct} / ${trainingSize} training examples correct · ${updates} weight updates`;
        ['weight-warmth', 'weight-brightness', 'weight-bias'].forEach((id, i) => {
            get(id).textContent = weights[i].toFixed(2);
        });
        get('learning-text').textContent = 'This scout is a perceptron: it adjusts its weights when it makes a training mistake. Training for a fixed number of passes does not guarantee a perfect model.';
        phase = 'testing';
        index = 0;
        showSignal();
    }

    function answer(choice) {
        const sample = samples[phase][index];
        const actual = sample.label === 1 ? 'Match' : 'Not a match';
        const playerCorrect = choice === sample.label;
        let message = `${playerCorrect ? 'You got it!' : 'Not quite.'} The correct answer is ${actual}.`;
        if (phase === 'training') {
            message += ' The scout will learn from the correct label.';
        } else {
            const prediction = predict(sample);
            playerScore += Number(playerCorrect);
            scoutScore += Number(prediction === sample.label);
            tested++;
            message += ` Scout predicted ${prediction === 1 ? 'Match' : 'Not a match'} — ${prediction === sample.label ? 'correct' : 'incorrect'}.`;
        }
        setFeedback(message);
        get('answer-actions').hidden = true;
        const last = index === (phase === 'training' ? trainingSize : testSize) - 1;
        const button = last ? get(phase === 'training' ? 'train-button' : 'next-mission-button') : get('continue-button');
        button.textContent = last && phase === 'testing' && missionIndex === missions.length - 1 ? 'See final results' : last && phase === 'testing' ? 'Next mission' : button.textContent;
        button.hidden = false;
        button.focus();
        updateDashboard();
    }

    function finishGame() {
        phase = 'complete';
        get('signal-card').hidden = true;
        get('question').hidden = true;
        get('answer-actions').hidden = true;
        get('next-mission-button').hidden = true;
        get('round-progress').textContent = 'All three missions complete';
        get('phase-hint').textContent = 'Challenge finished';
        get('mission-title').textContent = 'Mission complete!';
        get('mission-rule').textContent = 'You trained three scouts and tested each one on new signals.';
        setFeedback(`Final score: you ${playerScore} / ${tested}, scout ${scoutScore} / ${tested}. Restart to try a fresh set of signals.`);
        get('learning-text').textContent = 'A model can make mistakes on new data even when it performs well on training examples. Different examples can lead to different weights and results.';
        updateDashboard();
        get('reset-button').focus();
    }

    root.querySelectorAll('[data-answer]').forEach((button) => {
        button.addEventListener('click', () => answer(Number(button.dataset.answer)));
    });
    get('continue-button').addEventListener('click', () => {
        index++;
        showSignal();
        root.querySelector('[data-answer="1"]').focus();
    });
    get('train-button').addEventListener('click', () => {
        train();
        root.querySelector('[data-answer="1"]').focus();
    });
    get('next-mission-button').addEventListener('click', () => {
        if (missionIndex === missions.length - 1) {
            finishGame();
        } else {
            missionIndex++;
            startMission();
            root.querySelector('[data-answer="1"]').focus();
        }
    });
    get('reset-button').addEventListener('click', () => {
        missionIndex = 0;
        playerScore = 0;
        scoutScore = 0;
        tested = 0;
        startMission();
    });
    get('reset-button').click();
})();
