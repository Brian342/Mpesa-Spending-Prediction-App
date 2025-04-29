document.getElementById("prediction").addEventListener("submit", function(e) {
    e.preventDefault();

    const transactionAmount = document.getElementById("transactionAmount").value;
    const transactionType = document.getElementById("transactionType").value;
    const transactionParty = document.getElementById("transactionParty").value;
    const PaidInOrWithdrawal = document.getElementById("PaidInOrWithdrawal").value;

    if (!transactionAmount) {
        alert("Please enter a transaction amount!");
        return;
    }

    const data = {
        transactionAmount,
        transactionType,
        transactionParty,
        PaidInOrWithdrawal,
    }

    fetch('http://127.0.0.1:5000/prediction',{
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body:JSON.stringify(data)
    })

        .then(response => response.json())
        .then(prediction => {
            console.log(prediction);
            const resultContainer = document.getElementById("result");
            resultContainer.style.display = "block";
            resultContainer.innerHTML = `<h3>Predicted Spending: ${prediction.predicted_spending}</h3>`;
        })
        .catch(error => {
            console.error("Error:", error);
            alert("An error occurred. Please try again later.");
        });
});

