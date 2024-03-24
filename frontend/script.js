// Объект для хранения результатов
const modelResults = {};

async function trainModel(modelId) {
    const button = document.getElementById(modelId);
    button.classList.add('active');
  
    try {
        const response = await fetch(`http://localhost:8000/train_${modelId.toLowerCase()}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({}) // Если ваш API ожидает данные, добавьте их здесь
        });

        if (response.ok) {
            const result = await response.json();
            modelResults[modelId] = result; // Сохраняем результат для модели
            updateResults();
        } else {
            console.error('Сервер вернул ошибку:', response.statusText);
        }
    } catch (error) {
        console.error('Ошибка при обращении к серверу:', error);
    }
  
    button.classList.remove('active');
}

function updateResults() {
    const resultsDisplay = document.getElementById('results-display');
    resultsDisplay.innerHTML = ''; // Очищаем отображение результатов

    // Перебираем все результаты и добавляем их в отображение
    for (const [modelId, result] of Object.entries(modelResults)) {
        const resultText = document.createElement('div');
        resultText.textContent = `Results for ${modelId.toUpperCase()}: Loss - ${result.loss}, MAE - ${result.mae}`;
        resultsDisplay.appendChild(resultText);
    }
}
