import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import ClassifierMixin, MultiOutputMixin, BaseEstimator
from sklearn.metrics import accuracy_score


class TwoLayerNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_classes, method):
        super(TwoLayerNet, self).__init__()
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.method = method
        self.temperature = nn.Parameter(torch.ones(1))
        self.keys = nn.Parameter(torch.empty(n_hidden, n_input))
        self.values = nn.Parameter(torch.empty(n_hidden, n_classes))
        if method == "fc-relu":
            nn.init.kaiming_uniform_(self.keys, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.values, a=math.sqrt(5))
        else:
            nn.init.xavier_uniform_(self.keys)
            nn.init.xavier_uniform_(self.values)

    def forward(self, x):
        if self.method == "rbf":
            dist = torch.cdist(x, self.keys)
            attn = -1 * dist.pow(2)
            x = (attn / self.temperature).softmax(1) @ self.values
        elif self.method == "idw":
            # Add a diagonal transform to both x and keys
            dist = torch.cdist(x, self.keys)
            attn = 1 / (1e-4 + dist)
            x = (attn / self.temperature).softmax(1) @ self.values
        elif self.method == "dot":
            scaling = np.sqrt(1 / self.n_hidden)
            attn = scaling * x @ self.keys.T
            x = F.softmax(attn, 1) @ self.values
        elif self.method == "fc-relu":
            x = (x @ self.keys.T).relu() @ self.values
        return x


class TwoLayerNetClassifier(ClassifierMixin, MultiOutputMixin, BaseEstimator):
    def __init__(self, method, n_hidden=-1):
        self.net = None
        self.method = method
        self.n_hidden = n_hidden

    def fit(self, X, y):
        n, n_input = X.shape
        n_classes = len(np.unique(y))
        self.classes_ = np.unique(y)
        assert n_classes == np.max(y) + 1
        twolayernet = TwoLayerNet(n_input, self.n_hidden, n_classes, self.method)

        n_epochs = 20
        batch_size = 10
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(twolayernet.parameters(), lr=0.01)
        inputs = torch.from_numpy(X.astype(np.float32)).float()
        target = torch.from_numpy(y)
        for epoch in range(n_epochs):

            shuffle_idx = torch.randperm(inputs.size()[0])
            shuffled_input = inputs[shuffle_idx]
            shuffled_target = target[shuffle_idx]

            pred = []
            ground_truth = []
            total_loss = 0.

            for batch_idx in range(0, len(inputs), batch_size):
                sx = shuffled_input[batch_idx: batch_idx + batch_size]
                sy = shuffled_target[batch_idx: batch_idx + batch_size]

                # Forward pass
                output = twolayernet(sx)

                # Compute loss
                loss = criterion(output, sy)
                total_loss += loss.item()

                ground_truth.extend(sy.flatten().tolist())
                pred.extend(output.argmax(dim=-1).tolist())

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy = accuracy_score(ground_truth, pred)
            avg_loss = total_loss / (len(inputs) / batch_size)
            """
            if epoch % 5 == 0:
                print(f"  {self.method} {self.n_hidden} Epoch {epoch}, Accuracy {accuracy:.4f}, Loss {avg_loss:4f}")
            """
            if accuracy == 1.0:
                break
        #print(f"{self.method} {self.n_hidden} Epoch {epoch}, Accuracy {accuracy:.4f}, Loss {avg_loss:4f}")

        self.net = twolayernet
    
    def predict_proba(self, X):
        inputs = torch.from_numpy(X.astype(np.float32)).float()
        output = self.net(inputs).detach().softmax(-1).numpy()
        return output

    def predict(self, X):
        return self.predict_proba(X).argmax(dim=-1)
