# Tic-Tac-Toe with Minimax Algorithm
# Player can choose X or O

# -------------------------------
# Print the game board
# -------------------------------
def print_board(board):
    print(board[0], "|", board[1], "|", board[2])
    print("--+---+--")
    print(board[3], "|", board[4], "|", board[5])
    print("--+---+--")
    print(board[6], "|", board[7], "|", board[8])
    print()

# -------------------------------
# Check winner
# -------------------------------
def check_winner(board, player):
    win_positions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
        (0, 4, 8), (2, 4, 6)              # Diagonals
    ]

    for a, b, c in win_positions:
        if board[a] == board[b] == board[c] == player:
            return True
    return False

# -------------------------------
# Check draw
# -------------------------------
def is_draw(board):
    return " " not in board

# -------------------------------
# Player chooses symbol
# -------------------------------
def choose_symbol():
    while True:
        choice = input("Choose your symbol (X or O): ").upper()
        if choice == "X":
            return "X", "O"
        elif choice == "O":
            return "O", "X"
        else:
            print("Invalid choice. Please choose X or O.")

# -------------------------------
# Human move
# -------------------------------
def human_move(board, player):
    while True:
        move = int(input("Enter your move (0-8): "))
        if 0 <= move <= 8 and board[move] == " ":
            board[move] = player
            break
        else:
            print("Invalid move. Try again.")

# -------------------------------
# Minimax Algorithm
# -------------------------------
def minimax(board, is_maximizing, ai, player):

    if check_winner(board, ai):
        return 1
    if check_winner(board, player):
        return -1
    if is_draw(board):
        return 0

    if is_maximizing:  # AI turn
        best_score = -100
        for i in range(9):
            if board[i] == " ":
                board[i] = ai
                score = minimax(board, False, ai, player)
                board[i] = " "
                best_score = max(score, best_score)
        return best_score

    else:  # Human turn
        best_score = 100
        for i in range(9):
            if board[i] == " ":
                board[i] = player
                score = minimax(board, True, ai, player)
                board[i] = " "
                best_score = min(score, best_score)
        return best_score

# -------------------------------
# AI move
# -------------------------------
def ai_move(board, ai, player):
    best_score = -100
    best_move = 0

    for i in range(9):
        if board[i] == " ":
            board[i] = ai
            score = minimax(board, False, ai, player)
            board[i] = " "
            if score > best_score:
                best_score = score
                best_move = i

    board[best_move] = ai

# -------------------------------
# Main game loop
# -------------------------------
def play_game():
    board = [" " for _ in range(9)]
    player, ai = choose_symbol()

    while True:
        print_board(board)
        human_move(board, player)

        if check_winner(board, player):
            print_board(board)
            print("🎉 Congratulations! You win!")
            break

        if is_draw(board):
            print_board(board)
            print("🤝 It's a draw!")
            break

        ai_move(board, ai, player)

        if check_winner(board, ai):
            print_board(board)
            print("🤖 AI wins! Better luck next time.")
            break

# -------------------------------
# Start the game
# -------------------------------
play_game()
