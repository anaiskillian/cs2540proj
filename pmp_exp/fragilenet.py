import torch
import torch.nn as nn
import numpy as np
from z3 import Solver, Real, If as zIf, sat
import matplotlib.pyplot as plt

# overall summary: this NN is designed to be fragile due to conflicting data points and only partial training

class FragileNet(nn.Module):
    def __init__(self):
        super(FragileNet, self).__init__()
        # create TWO fragile decision boundaries
        self.fc1 = nn.Linear(2, 2)  # Two neurons
        self.fc2 = nn.Linear(2, 2)  # Output layer
        
        # Manually set fragile weights (training will slightly perturb these)
        with torch.no_grad():
            # First neuron: x - y > 0.1 (fragile at x-y≈0.1)
            self.fc1.weight[0] = torch.tensor([1.0, -1.0])
            self.fc1.bias[0] = torch.tensor([-0.1])
            
            # Second neuron: x + y > 0.9 (fragile at x+y≈0.9)
            self.fc1.weight[1] = torch.tensor([1.0, 1.0])
            self.fc1.bias[1] = torch.tensor([-0.9])
            
            # Output layer weights to create confusion
            self.fc2.weight[:] = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
            self.fc2.bias[:] = torch.tensor([0.0, 0.0])

    def forward(self, x):
        x = torch.sigmoid(50 * self.fc1(x))  # Sharp sigmoid approximates step
        return self.fc2(x)
    
    def get_params_for_smt(self):
        """Extract parameters for SMT encoding"""
        w1 = self.fc1.weight.detach().numpy()
        b1 = self.fc1.bias.detach().numpy()
        w2 = self.fc2.weight.detach().numpy()
        b2 = self.fc2.bias.detach().numpy()
        return w1, b1, w2, b2

def create_vulnerable_data():
    """Data that forces the network into fragile configurations"""
    # Points that will lie exactly on decision boundaries
    X = np.array([
        [0.10, 0.00],  # On x-y=0.1 boundary
        [0.45, 0.45],   # On x+y=0.9 boundary
        [0.20, 0.10],   # Just above x-y=0.1
        [0.50, 0.40],   # Just below x+y=0.9
        [0.30, 0.20],   # Between boundaries
        [0.60, 0.30]    # Between boundaries
    ], dtype=np.float32)
    
    y = np.array([1, 1, 0, 1, 0, 1], dtype=np.int64)  # Contradictory labels
    return X, y

def train_with_controlled_fragility(model, X, y, epochs=200):
    """Training that preserves designed vulnerabilities (so only partial training)"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X))
        loss = criterion(outputs, torch.LongTensor(y))
        loss.backward()
        
        # Freeze gradients for the fragile directions
        with torch.no_grad():
            model.fc1.weight[0][0] -= 0.001 * model.fc1.weight.grad[0][0]  # Only partial updates
            model.fc1.weight[0][1] += 0.001 * model.fc1.weight.grad[0][1]
            model.fc1.bias[0] += 0.001 * model.fc1.bias.grad[0]
            
            model.fc1.weight[1] += 0.001 * model.fc1.weight.grad[1] 
        optimizer.step()

def encode_network_in_z3(model, x_vars):
    """Encode the trained network in Z3 constraints"""
    w1, b1, w2, b2 = model.get_params_for_smt()
    
    constraints = []
    out1 = Real("out1")
    out2 = Real("out2")
    expr1 = w1[0,0] * x_vars[0] + w1[0,1] * x_vars[1] + b1[0]
    expr2 = w1[1,0] * x_vars[0] + w1[1,1] * x_vars[1] + b1[1]
    constraints.append(out1 == expr1)
    constraints.append(out2 == expr2)
    
    # Sigmoid activation (approximated as step)
    a1 = Real("a1")
    a2 = Real("a2")
    constraints.append(a1 == zIf(out1 > 0, 1.0, 0.0))
    constraints.append(a2 == zIf(out2 > 0, 1.0, 0.0))
    
    # Output layer
    final_out0 = Real("final_out0")
    final_out1 = Real("final_out1")
    expr_out0 = a1 * w2[0,0] + a2 * w2[0,1] + b2[0]
    expr_out1 = a1 * w2[1,0] + a2 * w2[1,1] + b2[1]
    constraints.append(final_out0 == expr_out0)
    constraints.append(final_out1 == expr_out1)
    
    return constraints, [final_out0, final_out1]

def verify_robustness(model, x_sample, eps, exp_class):
    """Verify if there exists an adversarial example within epsilon"""
    solver = Solver()
    xv = [Real(f"x_{i}") for i in range(len(x_sample))]
    xv_pert = [Real(f"x_pert_{i}") for i in range(len(x_sample))]
    
    for i in range(len(x_sample)):
        solver.add(xv[i] == float(x_sample[i]))
        solver.add(xv_pert[i] >= float(x_sample[i]) - eps)
        solver.add(xv_pert[i] <= float(x_sample[i]) + eps)
    
    c_pert, out_pert = encode_network_in_z3(model, xv_pert)
    for c in c_pert:
        solver.add(c)
    
    if exp_class == 0:
        solver.add(out_pert[1] > out_pert[0])  
    else:
        solver.add(out_pert[0] > out_pert[1]) 
    
    return solver.check() == sat

def gradient_attack(model, x_sample, eps, exp_class, steps=30):
    """Simple gradient-based attack"""
    x_t = torch.FloatTensor(x_sample).clone()
    x_t.requires_grad = True
    target = torch.LongTensor([1 - exp_class])
    ce = nn.CrossEntropyLoss()
    
    for _ in range(steps):
        logits = model(x_t.unsqueeze(0))
        loss = -ce(logits, target)
        loss.backward()
        
        if x_t.grad is None:
            return False 
        
        with torch.no_grad():
            x_t += (eps / steps) * x_t.grad.sign()
            x_t = torch.clamp(x_t, torch.FloatTensor(x_sample) - eps, 
                             torch.FloatTensor(x_sample) + eps)
            x_t.requires_grad = True
            
            pred = model(x_t.unsqueeze(0)).argmax().item()
            if pred != exp_class:
                return True
        
        x_t.grad = None
    
    return False

def run_experiment():
    X, y = create_vulnerable_data()
    model = FragileNet()
    train_with_controlled_fragility(model, X, y)
    
    # Test points specifically on the decision boundaries
    test_points = [
        ([0.10, 0.00], 0),  # On x-y=0.1 boundary
        ([0.45, 0.45], 1),   # On x+y=0.9 boundary
        ([0.20, 0.10], 0),   # Just above x-y=0.1
        ([0.50, 0.40], 1)    # Just below x+y=0.9
    ]
    
    eps_list = [0.001, 0.005, 0.01, 0.02]
    print("Testing vulnerabilities at specific points:")
    
    for eps in eps_list:
        smt_vuln = grad_vuln = 0
        for x, true_class in test_points:
            with torch.no_grad():
                pred = model(torch.FloatTensor(x).unsqueeze(0)).argmax().item()
            
            if pred == true_class: 
                if verify_robustness(model, x, eps, pred):
                    smt_vuln += 1
                if gradient_attack(model, x, eps, pred):
                    grad_vuln += 1
        
        print(f"ε={eps:.3f}: SMT found {smt_vuln}/4 vulnerable, "
              f"Gradient found {grad_vuln}/4 vulnerable")

if __name__ == "__main__":
    run_experiment()