function openChart(evt, chartName) {
    console.log('Opening chart:', chartName); // Debug log
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(chartName).style.display = "block";
    evt.currentTarget.className += " active";
}

// Show the first tab by default and handle form submission
document.addEventListener("DOMContentLoaded", function() {
    console.log('DOM loaded, initializing tabs and form'); // Debug log
    // Open the first tab by default
    const firstTab = document.getElementsByClassName("tablinks")[0];
    if (firstTab) {
        firstTab.click();
    } else {
        console.warn('No tablinks found'); // Debug log
    }
    
    // Handle future prediction form submission
    const predictForm = document.getElementById('predictFutureForm');
    if (predictForm) {
        console.log('Found predictFutureForm, attaching submit handler'); // Debug log
        predictForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const formData = new FormData(predictForm);
            const data = {
                stock_symbol: formData.get('stock_symbol'),
                target_date: formData.get('target_date')
            };
            console.log('Sending AJAX request with data:', data); // Debug log
            
            fetch('/api/predict_future', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => {
                console.log('Raw response:', response); // Debug log
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(result => {
                console.log('Received response:', result); // Debug log
                const resultDiv = document.getElementById('futurePredictionResult');
                if (result.error) {
                    console.error('Server returned error:', result.error); // Debug log
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
                } else {
                    resultDiv.innerHTML = `
                        <p><strong>Prediction for ${result.stock_symbol} on ${result.target_date}</strong></p>
                        <p>Predicted Price: $${result.predicted_price.toFixed(2)}</p>
                        <p>Price Change: ${result.price_change >= 0 ? '+' : ''}$${result.price_change.toFixed(2)} 
                           (${result.price_change_percent.toFixed(2)}%)</p>
                        <p>Confidence: ${result.confidence}</p>
                        <p>Recommendation: ${result.recommendation}</p>
                        <p>Trend: ${result.trend}</p>
                    `;
                }
            })
            .catch(error => {
                console.error('AJAX error:', error); // Debug log
                document.getElementById('futurePredictionResult').innerHTML = 
                    `<p style="color: red;">Error: ${error.message}</p>`;
            });
        });
    } else {
        console.warn('predictFutureForm not found'); // Debug log
    }
});