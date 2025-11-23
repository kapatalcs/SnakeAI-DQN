# Snake AI - Deep Reinforcement Learning (DQN + Hybrid Solver)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Pygame](https://img.shields.io/badge/Pygame-Environment-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A self-learning AI that plays the classic Snake game using Deep
> Q-Learning combined with a rule-based Hybrid Solver.

------------------------------------------------------------------------

## 🐍 Project Overview

This project trains an artificial intelligence agent to play the classic
Snake game using:

-   **Deep Q-Learning (DQN)**
-   **Hybrid rule‑based safety logic**
-   **Replay Memory (Experience Replay)**
-   **Linear Q‑Network (PyTorch)**

The agent:

-   Observes the environment (walls, food, tail).
-   Selects the action with the highest expected reward.
-   Hybrid Solver prevents suicidal or trapped moves.
-   Learns from past mistakes and can reach **200+ scores**.

------------------------------------------------------------------------

## 🎯 Why Hybrid Solver?

Pure DQN behaves randomly early in training, causing frequent crashes
such as:

-   Hitting a wall,
-   Trapping itself inside its own tail,
-   Entering dead‑end states.

The **Hybrid Solver** mitigates these issues through:

-   Wall‑hit pre‑checks,
-   Tail‑trap detection,
-   Safe‑turn heuristics,
-   Basic risk‑avoidance logic.

This stabilizes learning and significantly speeds up performance
improvement.

------------------------------------------------------------------------

## 📁 Project Structure

     Snake-AI
     ┣ agent.py        # DQN, Replay Memory, Hybrid Solver, training loop
     ┣ model.py        # PyTorch Linear Q‑Network
     ┣ gameAI.py       # Pygame Snake environment
     ┣ analyzer.py     # Score & mean‑score plotting tool
     ┗ requirements.txt

------------------------------------------------------------------------

## ⚙️ Installation

> Requires **Python 3.10+**

``` bash
git clone https://github.com/kullanici/Snake-AI.git
cd Snake-AI
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 Usage

Start training:

``` bash
python agent.py
```

During training:

-   Console prints **Score**, **Record**, and **Mean Score**.
-   A real‑time graph displays the learning curve.
-   The game runs in accelerated mode.

### Stop Training

Press:

    CTRL + C

The best model is automatically saved as:

    model/model.pth

------------------------------------------------------------------------

## 🔧 Configuration

Editable inside `agent.py`:

  Variable          Description              Default
  ----------------- ------------------------ ---------
  `MAX_MEMORY`      Replay buffer size       200,000
  
  `BATCH_SIZE`      Number of samples/step   2048
  
  `LEARNING_RATE`   Learning rate (LR)       0.001

Game speed can be modified inside `gameAI.py` via `SPEED`.

------------------------------------------------------------------------

## 📬 Contact

Feel free to reach out for contributions, suggestions, or improvements.

------------------------------------------------------------------------

Enjoy training and high scoring! 🐍🤖
