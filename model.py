import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import os
import numpy

class Linear_QNET(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, hiddenSize)
        self.linear3 = nn.Linear(hiddenSize, hiddenSize//2)
        self.linear4 = nn.Linear(hiddenSize//2, outputSize)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0)

    def forward(self,x):
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x


class QTrainer:
    def __init__(self, model, targetModel,learningRate, gamma):
        self.learningRate = learningRate
        self.gamma = gamma
        self.model = model
        self.targetModel = targetModel
        self.optimizer = optim.Adam(model.parameters(), lr = self.learningRate)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, next_state, done):
        state = torch.tensor(numpy.array(state), dtype=torch.float)
        next_state = torch.tensor(numpy.array(next_state), dtype=torch.float)
        action = torch.tensor(numpy.array(action), dtype=torch.float)
        reward = torch.tensor(numpy.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        prediction = self.model(state)
        next_actions_main = self.model(next_state).detach()
        best_actions_indices = torch.argmax(next_actions_main, dim=1)

        next_state_values_target = self.targetModel(next_state).detach()

        target = prediction.clone()
        for indx in range(len(done)):
            Q_new = reward[indx]
            if not done[indx]:
                selected_action_idx = best_actions_indices[indx]
                Q_new = reward[indx] + self.gamma * next_state_values_target[indx][selected_action_idx]

            target[indx][torch.argmax(action[indx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target,prediction)
        loss.backward()

        self.optimizer.step()
