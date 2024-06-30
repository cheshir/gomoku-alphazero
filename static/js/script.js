document.addEventListener("DOMContentLoaded", () => {
    const boardElement = document.getElementById("board");
    const statusElement = document.getElementById("status");
    let currentPlayer = 1;
    let winnerPlayer = 0;

    function createBoard() {
        boardElement.innerHTML = '';
        for (let i = 0; i < 15; i++) {
            for (let j = 0; j < 15; j++) {
                const cell = document.createElement("div");
                cell.className = "cell";
                cell.dataset.row = i;
                cell.dataset.col = j;
                cell.addEventListener("click", onCellClick);
                boardElement.appendChild(cell);
            }
        }
    }

    function onCellClick(event) {
        if (winnerPlayer) {
            return;
        }

        const row = event.target.dataset.row;
        const col = event.target.dataset.col;
        makeMove(row, col);
    }

    function makeMove(row, col) {
        axios.post("/move", {
            row: parseInt(row),
            col: parseInt(col),
            current_player: currentPlayer
        })
        .then(response => {
            const data = response.data;
            if (data.error) {
                alert(data.error);
            } else {
                updateBoard(data.board);
                currentPlayer = data.current_player;
                updateStatus(data.winner);
            }
        })
        .catch(error => console.error("Error:", error));
    }

    function updateBoard(board) {
        const cells = document.querySelectorAll(".cell");
        cells.forEach(cell => {
            const row = cell.dataset.row;
            const col = cell.dataset.col;
            const value = board[row][col];
            if (value === 1) {
                cell.textContent = "X";
                cell.style.color = "#ff0000";  // Red color for Player 1
            } else if (value === 2) {
                cell.textContent = "O";
                cell.style.color = "#0000ff";  // Blue color for Player 2
            } else {
                cell.textContent = "";
            }
        });
    }

    function updateStatus(winner) {
        winnerPlayer = winner;

        if (winner) {
            statusElement.textContent = `Player ${winner} wins!`;
        } else {
            statusElement.textContent = `Player ${currentPlayer}'s turn`;
        }
    }

    function resetGame() {
        axios.post("/reset")
        .then(response => {
            const data = response.data;
            updateBoard(data.board);
            currentPlayer = data.current_player;
            statusElement.textContent = `Player ${currentPlayer}'s turn`;
        })
        .catch(error => console.error("Error:", error));
    }

    createBoard();
    resetGame();

    window.resetGame = resetGame; // Expose the function to the global scope
});