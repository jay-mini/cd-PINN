# -*- coding: utf-8 -*-
# @Time    : 2025/9/2 15:52
# @Author  : Jay
# @File    : cd_PINN.py
# @Project: CNO
import torch
import numpy as np
from matplotlib import pyplot as plt
from utils import set_seed, FCNN
import time
from choose_optimizer import choose_optimizer
import skopt
from torch.optim.lr_scheduler import StepLR


class PICNO(torch.nn.Module):
    def __init__(self, X_u_train, u_train, X_f_train, layers, lr, optimizer_name, iterations):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.t_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).float().to(self.device)
        self.x_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True).float().to(self.device)
        self.y_u = torch.tensor(X_u_train[:, 2:3], requires_grad=True).float().to(self.device)
        self.z_u = torch.tensor(X_u_train[:, 3:4], requires_grad=True).float().to(self.device)
        self.t_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(self.device)
        self.x_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(self.device)
        self.y_f = torch.tensor(X_f_train[:, 2:3], requires_grad=True).float().to(self.device)
        self.z_f = torch.tensor(X_f_train[:, 3:4], requires_grad=True).float().to(self.device)
        self.t_0 = torch.tensor(0.1 * np.ones_like(X_f_train[:, 0:1]), requires_grad=True).float().to(self.device)
        self.u = torch.tensor(u_train, requires_grad=True).float().to(self.device)

        self.net = FCNN(layers)
        self.net.to(self.device)
        self.k = 0.2
        self.sigma1 = 1.0
        self.mu1 = 5.0

        self.optimizer = choose_optimizer(optimizer_name, self.net.parameters(), lr)
        self.optimizer1 = choose_optimizer('LBFGS', self.net.parameters())
        self.iterations = iterations
        self.scheduler = StepLR(self.optimizer, step_size=5000, gamma=0.99)
        self.iter = 0
        self.loss, self.loss_u, self.loss_f, self.loss_u0, self.loss_b, self.loss_c, self.loss_c0 = [], [], [], [], [], [], []

    def net_u(self, x, t, y, z):
        u = self.net(torch.cat([x, t, y, z], dim=1))
        return u

    def net_f(self, x, t, y, z):
        u = self.net_u(x, t, y, z)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        u_ty = torch.autograd.grad(u_t, y, grad_outputs=torch.ones_like(u_t), create_graph=True, retain_graph=True)[
                0]
        u_tz = torch.autograd.grad(u_t, z, grad_outputs=torch.ones_like(u_t), create_graph=True, retain_graph=True)[
                0]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[
                0]
        u_xxy = \
                torch.autograd.grad(u_xx, y, grad_outputs=torch.ones_like(u_xx), create_graph=True, retain_graph=True)[
                    0]
        u_xxz = \
                torch.autograd.grad(u_xx, z, grad_outputs=torch.ones_like(u_xx), create_graph=True, retain_graph=True)[
                    0]

        f1 = u_t - self.k * u_xx
        f2 = u_ty - self.k * u_xxy
        f3 = u_tz - self.k * u_xxz

        return f1, f2, f3

    def net_c0(self, x, t, y, z):
        u = self.net_u(x, t, y, z)
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        phi_y = -10 * y / ((y ** 2 + 0.2 * self.k) * torch.sqrt(
            torch.tensor(2 * torch.pi) * (y ** 2 + 0.2 * self.k))) * torch.exp(
            -(x - z) ** 2 / (2 * (y ** 2 + 0.2 * self.k))) + 10 * y * (x - z) ** 2 / (
                        (y ** 2 + 0.2 * self.k) ** 2 * torch.sqrt(
                    torch.tensor(2 * torch.pi) * (y ** 2 + 0.2 * self.k))) * torch.exp(
            -(x - z) ** 2 / (2 * (y ** 2 + 0.2 * self.k)))
        phi_z = 10 * (x - z) / ((y ** 2 + 0.2 * self.k) * torch.sqrt(
            torch.tensor(2 * torch.pi) * (y ** 2 + 0.2 * self.k))) * torch.exp(
            -(x - z) ** 2 / (2 * (y ** 2 + 0.2 * self.k)))

        return u_y - phi_y, u_z - phi_z

    def net_u0(self, x, t, y, z):
        u = self.net_u(x, t, y, z)

        phi = 10 / (torch.sqrt(torch.tensor(2 * torch.pi) * (y ** 2 + 0.2 * self.k))) * torch.exp(
            -(x - z) ** 2 / (2 * (y ** 2 + 0.2 * self.k))) + 10 / (
                    torch.sqrt(torch.tensor(2 * torch.pi * (self.sigma1 ** 2 + 0.2 * self.k)))) * torch.exp(
            torch.tensor(-(x - self.mu1) ** 2 / (2 * (self.sigma1 ** 2 + 0.2 * self.k))))

        return u - phi

    def loss(self, verbose=True):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
            self.optimizer1.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u, self.y_u, self.z_u)
        f1, f2, f3 = self.net_f(self.x_f, self.t_f, self.y_f, self.z_f)
        u0 = self.net_u0(self.x_f, self.t_0, self.y_f, self.z_f)
        uy0, uz0 = self.net_c0(self.x_f, self.t_0, self.y_f, self.z_f)

        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f1 ** 2)
        loss_c = torch.mean(f2 ** 2) + torch.mean(f3 ** 2)
        loss_u0 = torch.mean(u0 ** 2)
        loss_c0 = torch.mean(uy0 ** 2) + torch.mean(uz0 ** 2)

        loss = loss_u + loss_f + loss_u0 + loss_c + loss_c0

        if loss.requires_grad:
            loss.backward()

        grad_norm = 0
        for p in self.net.parameters():
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        if verbose:
            if self.iter % 100 == 0:
                print(
                    'epoch %d, gradient: %.5e, loss:%.5e, loss_u: %.5e, loss_f: %.5e, loss_u0: %.5e, loss_c: %.5e, loss_c0: %.5e' % (
                        self.iter, grad_norm, loss.item(), loss_u.item(), loss_f.item(), loss_u0.item(),
                        loss_c.item(), loss_c0.item()))
            self.iter += 1
            self.loss.append(loss.cpu().detach().item())
            self.loss_u.append(loss_u.cpu().detach().item())
            self.loss_f.append(loss_f.cpu().detach().item())
            self.loss_u0.append(loss_u0.cpu().detach().item())
            self.loss_c.append(loss_c.cpu().detach().item())
            self.loss_c0.append(loss_c0.cpu().detach().item())

        return loss

    def train(self):
        self.net.train()
        for i in range(self.iterations):
            self.optimizer.step(self.loss_pinn)
        print("The training process using adam is finished!")
        self.optimizer1.step(self.loss_pinn)

        return self.loss, self.loss_u, self.loss_f, self.loss_u0, self.loss_c, self.loss_c0

    def predict(self, X):
        t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        x = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        y = torch.tensor(X[:, 2:3], requires_grad=True).float().to(self.device)
        z = torch.tensor(X[:, 3:4], requires_grad=True).float().to(self.device)

        self.net.eval()
        u = self.net_u(x, t, y, z)
        u = u.detach().cpu().numpy()

        return u


if __name__ == '__main__':
    set_seed(0)
    iterations = 50000
    N_f = 8192 * 2
    lb = np.array([0.0, -10.0, 0.1, -5.0])
    ub = np.array([1.0, 10.0, 10.0, 5.0])
    optimizer_name = 'Adam'
    lr = 1e-3
    path = r'../Data'
    file_train = r'/training_data.npz'
    file_test = r'/testing_data.npz'
    data_test = np.load(path + file_test)
    data_train = np.load(path + file_train)
    X_train, U_train, u_train = data_train['arr1'], data_train['arr2'], data_train['arr3']
    X_test, U_test, u_test = data_test['arr1'], data_test['arr2'], data_test['arr3']

    space = [(lb[0], ub[0]), (lb[1], ub[1]), (lb[2], ub[2]), (lb[3], ub[3])]
    sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    X_f_train = np.array(sampler.generate(space, N_f))
    X_u_train = np.array(X_train[::5])
    U_train1 = np.array(U_train[::5])

    layers = [4, 128, 128, 128, 128, 128, 128, 1]
    model = PICNO(X_u_train, U_train1, X_f_train, layers, lr, optimizer_name, iterations)
    start_time = time.time()
    # loss, loss_u, loss_f, loss_u0, loss_c, loss_c0 = model.train()
    end_time = time.time()
    model.load_state_dict(torch.load(path+r'/PINN.pth'))

    u_train_pred = model.predict(X_train)
    u_test_pred = model.predict(X_test)

    error_u_NRMSE_train = np.linalg.norm(u_train_pred - U_train, 2) / np.linalg.norm(U_train.reshape(-1, 1), 2)
    error_u_NRMSE_test = np.linalg.norm(u_test_pred - U_test, 2) / np.linalg.norm(U_test, 2)
    print(f'Test Error: {error_u_NRMSE_test:.5e}, Train Error: {error_u_NRMSE_train:.5e}')
