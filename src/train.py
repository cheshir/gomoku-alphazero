import torch
import torch.optim as optim
from network import GomokuNet, create_data_loader
from self_play import self_play
import os
import glob

def get_latest_model(model_dir="models"):
    """Find the latest model in the models directory."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_files = glob.glob(os.path.join(model_dir, "model_iteration_*.pth"))
    if not model_files:
        return None
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model

def train_iteration(model, num_games=100, mcts_simulations=800, epochs=10, batch_size=32, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Generating self-play data...")
    training_data = self_play(model, num_games=num_games, mcts_simulations=mcts_simulations)

    print("Training model...")
    model.to(device)
    model.train()

    board_states, policies, values = zip(*training_data)
    data_loader = create_data_loader(board_states, policies, values, batch_size)

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            board_states_batch, policy_targets, value_targets = [item.to(device) for item in batch]

            optimizer.zero_grad()
            policy_output, value_output = model(board_states_batch)

            loss = model.loss(policy_output, value_output, policy_targets, value_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")

    print("Training completed")
    return model

def train(
    model,
    start_iteration=1,
    total_iterations=10,
    games_per_iteration=100,
    mcts_simulations=800,
    epochs=5,
    batch_size=32,
    device='cpu'
):
    for i in range(start_iteration, start_iteration + total_iterations):
        print(f"Training iteration {i}")
        model = train_iteration(
            model,
            num_games=games_per_iteration,
            mcts_simulations=mcts_simulations,
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )

        # Save model after each iteration
        model.save_model(f"models/model_iteration_{i}.pth")

    print("Training complete!")
    return model

def main():
    board_size = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    latest_model = get_latest_model()
    if latest_model:
        print(f"Continuing training from {latest_model}")
        model = GomokuNet(board_size=board_size)
        model.load_model(latest_model)
        start_iteration = int(latest_model.split('_')[-1].split('.')[0]) + 1
    else:
        print("Starting new training")
        model = GomokuNet(board_size=board_size)
        start_iteration = 1

    total_iterations = 10  # Number of training iterations
    model = train(
        model,
        start_iteration=start_iteration,
        total_iterations=total_iterations,
        games_per_iteration=100,
        mcts_simulations=800,
        epochs=5,
        batch_size=32,
        device=device
    )


if __name__ == "__main__":
    main()