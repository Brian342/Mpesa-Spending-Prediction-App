document.getElementById("predictBtn").addEventListener("click", function() {
    const transactionAmount = document.getElementById("transactionAmount").value;

    if (!transactionAmount) {
        alert("Please enter a transaction amount!");
        return;
    }

    fetch(`http://127.0.0.1:5000/prediction/${transactionAmount}`)
        .then(response => response.json())
        .then(data => {
            const resultContainer = document.getElementById("result");
            resultContainer.style.display = "block";
            resultContainer.innerHTML = `<h3>Predicted Balance: ${data.remaining_balance}</h3>`;
        })
        .catch(error => {
            console.error("Error:", error);
            alert("An error occurred. Please try again later.");
        });
});

