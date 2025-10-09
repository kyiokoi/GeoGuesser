// Country flag emojis
const countryFlags = {
    'Indonesia': '🇮🇩',
    'Laos': '🇱🇦',
    'Malaysia': '🇲🇾',
    'Philippines': '🇵🇭',
    'Singapore': '🇸🇬',
    'Thailand': '🇹🇭'
};

// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const predictBtn = document.getElementById('predictBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const resultsSection = document.getElementById('resultsSection');

let selectedFile = null;

// Upload box click
uploadBox.addEventListener('click', () => {
    fileInput.click();
});

// File selection
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

// Handle file
function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
    }

    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        predictBtn.disabled = false;
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Remove image
removeBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    predictBtn.disabled = true;
    resultsSection.style.display = 'none';
});

// Predict button
predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Show loading state
    btnText.textContent = 'Analyzing...';
    btnLoader.style.display = 'inline-block';
    predictBtn.disabled = true;

    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        // Send to backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        alert('Error making prediction. Please try again.');
    } finally {
        // Reset button
        btnText.textContent = 'Predict Country';
        btnLoader.style.display = 'none';
        predictBtn.disabled = false;
    }
});

// Display results
function displayResults(data) {
    const topPrediction = data.predictions[0];
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Update top prediction
    document.getElementById('countryFlag').textContent = countryFlags[topPrediction.country] || '🏳️';
    document.getElementById('countryName').textContent = topPrediction.country;
    document.getElementById('confidencePercent').textContent = topPrediction.confidence.toFixed(1);
    
    // Animate confidence bar
    const confidenceFill = document.getElementById('confidenceFill');
    setTimeout(() => {
        confidenceFill.style.width = topPrediction.confidence + '%';
    }, 100);

    // Display all predictions
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';
    
    data.predictions.forEach(pred => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <span class="prediction-item-country">
                ${countryFlags[pred.country] || '🏳️'} ${pred.country}
            </span>
            <span class="prediction-item-confidence">${pred.confidence.toFixed(1)}%</span>
        `;
        predictionsList.appendChild(item);
    });
}
