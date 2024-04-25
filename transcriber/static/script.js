'use-strict';

// Select elements
const backendSelector = document.getElementById('backend');
const modelSizeSelector = document.getElementById('model_size');
const deviceSelector = document.getElementById('device');

const samplerateSelector = document.getElementById('samplerate');
const blocksizeSelector = document.getElementById('blocksize');
const memorysafeSelector = document.getElementById('memory_safe');

const startButton = document.getElementById('startButton');
const outputField = document.getElementById('output');

let selectedBackend, selectedModelSize, selectedDevice, selectedSamplerate, selectedBlocksize, selectedMemorysafe;

const start = function() {
    selectedBackend = backendSelector.options[backendSelector.selectedIndex].value;
    selectedModelSize = modelSizeSelector.options[modelSizeSelector.selectedIndex].value;
    selectedDevice = deviceSelector.options[deviceSelector.selectedIndex].value;

    selectedSamplerate = samplerateSelector.value;
    selectedBlocksize = blocksizeSelector.value;
    selectedMemorysafe = memorysafeSelector.options[memorysafeSelector.selectedIndex].value;

    outputField.append("Started transcriber...\n");
    
    console.log("Started");
    startButton.removeEventListener("click", start);
    startButton.addEventListener("click", stop);
    startButton.textContent = "Stop";
};

const stop = function() {
    console.log("Stopped");

    outputField.append("Stopped transcriber...\n");
    
    startButton.removeEventListener("click", stop);
    startButton.addEventListener("click", start);
    startButton.textContent = "Start";
};

startButton.addEventListener('click', start);