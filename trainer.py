import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

class ALPaCA(torch.nn.Module):
    def __init__(self):
        super(ALPaCA, self).__init__()
        hidden_dim = 64
        #encoder network uses tanh as sinusoid prediction problem is bounded
        self.phi = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, hidden_dim))

        self.K_0 = torch.nn.Parameter(torch.zeros(hidden_dim, 1))
        self.L_0 = torch.nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.noise_parameter = 0.01
        #set A_0 = torch.matmul(self.L_0, self.L_0.T)
        #This enforces PSD

        self.reset_parameters()

    def reset_parameters(self):
        #torch.nn.init.xavier_uniform_(self.phi[0].weight)
        torch.nn.init.xavier_uniform_(self.K_0)
        torch.nn.init.xavier_uniform_(self.L_0)

    def forward(self, X_context, Y_context, x):
        #X_context is n_batches x len_context x 1
        #Y_context is n_batches x len_context x 1
        #x is n_batches x len_train_batch x 1, the example we backprop on

        context_phi = self.phi(X_context) #n_batches x len_context x hidden_dim
        phi_x = self.phi(x) #n_batches x len_train_batch x hidden_dim

        #derive posterior parameters
        A_0 = torch.matmul(self.L_0, self.L_0.T)
        A_context = torch.matmul(context_phi.transpose(1, 2), context_phi) + A_0
        factor = torch.matmul(context_phi.transpose(1, 2), Y_context) + torch.matmul(A_0, self.K_0) #breaks up calculation of K_context
        K_context = torch.matmul(torch.inverse(A_context), factor) #(n_batches, hidden_dim, hidden_dim) x (n_batches, hidden_dim, 1) = (n_batches, hidden_dim, 1)
        sigma_x = 1 + torch.matmul(phi_x, torch.matmul(torch.inverse(A_context), phi_x.transpose(1, 2))) #(n_batches, len_train_batch, len_train_batch)
        sigma_x = sigma_x * self.noise_parameter

        #calculate predictions for backprop examples
        preds = torch.matmul(phi_x, K_context) #preds is (n_batches, len_train_batch, 1)

        return preds, sigma_x

    def start_online(self):
        #computes and saves A^{-1} and Q
        A_0 = torch.matmul(self.L_0, self.L_0.T)
        self.Q = torch.matmul(A_0, self.K_0)
        self.A_inverse = torch.inverse(A_0)

    def online(self, x, update=False, y=None):
        #x and y are both of size 1
        phi_x = self.phi(x).unsqueeze(1) #(hidden_dim, 1)
        K = torch.matmul(self.A_inverse, self.Q)
        pred = torch.matmul(K.transpose(0, 1), phi_x)
        sigma = 1 + torch.matmul(phi_x.transpose(0, 1), torch.matmul(self.A_inverse, phi_x))
        sigma = sigma * self.noise_parameter
        if update:
            if y is None:
                raise Exception("When doing online updates, you must specify y")
            normalizer = 1 + torch.matmul(phi_x.transpose(0, 1), torch.matmul(self.A_inverse, phi_x))
            update_matrix = torch.matmul(torch.matmul(self.A_inverse, phi_x), torch.matmul(self.A_inverse, phi_x).transpose(0, 1))
            self.A_inverse = self.A_inverse - (1 / normalizer) * update_matrix
            self.Q = self.Q + phi_x * y
        return pred, sigma

def loss_fn(preds, sigma_x, actual_values):
    #preds is (n_batches, len_train_batch, 1)
    #sigma_x is (n_batches, len_train_batch, len_train_batch)
    #actual_values is (n_batches, len_train_batch, 1)
    difference = preds - actual_values
    log_gaussian = torch.matmul(difference.transpose(1, 2), torch.matmul(torch.inverse(sigma_x), difference)) #(n_batches, 1, 1)
    #print(log_gaussian)
    #print(sigma_x)
    log_det = torch.log(torch.linalg.det(sigma_x))
    log_det = log_det.unsqueeze(1).unsqueeze(2)
    #print(log_det)
    task_losses = log_det + log_gaussian
    task_losses = task_losses.squeeze(2)
    return torch.mean(task_losses, dim=0)

if __name__ == "__main__":
    model = ALPaCA()

    #data generation for offline initialization
    X_context = []
    Y_context = []
    x_train = []
    y_train = []
    n_batches = 3
    len_context = 20
    len_train_batch = 1 #in practice, exceeding 4 results in sigma_x being singular (should I use identity matrix instead of 1?)
    #probably is a better way to implement next 9 lines but oh well
    freqs = np.linspace(0.5, 0.5, 2*n_batches)
    for i in range(1, 2*n_batches + 1):
        X_context.append(torch.linspace(0, 1, len_context))
        Y_context.append(torch.sin(freqs[i-1] * 2 * np.pi * torch.linspace(0, 1, len_context)))
        x_train.append(torch.linspace(1, 1.2, len_train_batch))
        y_train.append(torch.sin(freqs[i-1] * 2 * np.pi * torch.linspace(1, 1.2, len_train_batch)))
    X_context = torch.stack(X_context).unsqueeze(2)
    Y_context = torch.stack(Y_context).unsqueeze(2)
    x_train = torch.stack(x_train).unsqueeze(2)
    y_train = torch.stack(y_train).unsqueeze(2)

    #quick sanity checks
    print(X_context.shape)
    print(Y_context.shape)
    print(x_train.shape)
    print(y_train.shape)

    #offline meta learning
    max_train_iter = 500 #for len_train_batch=1 doing more iterations than this doesn't change things in terms of loss
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    perturb_training_data = True
    training_losses_offline = []
    average_variances = []
    for iter_count in range(max_train_iter):
        #training step
        optimizer.zero_grad()
        preds, variances = model(X_context, Y_context, x_train)
        loss = loss_fn(preds, variances, y_train)
        loss.backward()
        optimizer.step()

        #for creating plots
        training_losses_offline.append(loss.item())
        average_variances.append(torch.mean(variances).detach().numpy())

        #for backpropping on different examples
        if perturb_training_data:
            #x_train = []
            y_train = []
            Y_context = []
            x_train = torch.rand((n_batches, len_train_batch, 1)) #experiment with this factor
            X_context = torch.rand((n_batches, len_context, 1))
            i = 0
            for iter in np.random.randint(n_batches*2, size=n_batches):
                y_train.append(torch.sin(freqs[iter] * 2 * np.pi * x_train[i]))#can't think of a better way to do this
                Y_context.append(torch.sin(freqs[iter] * 2 * np.pi * X_context[i]))
                i += 1
            y_train = torch.stack(y_train)
            Y_context = torch.stack(Y_context)
            #print(x_train.shape)
            #print(y_train.shape)
            #raiseException()
        if iter_count % 25 == 0:
            plt.figure((iter_count/100)+1)
            time.sleep(2)
            #Online Part
            #First want to see what the model returns when predicting over the interval (what the offline posterior parameters return)
            model.start_online()
            x = np.linspace(0, 1, 100)
            y_top = []
            y = []
            y_bottom = []
            for i in range(len(x)):
                pred, sigma = model.online(torch.tensor(x[i:i+1]).float())
                u = pred.detach().numpy()[0][0]
                v = np.sqrt(sigma.detach().numpy()[0][0]) #standard deviation
                y.append(u)
                y_bottom.append(u - 2*v)
                y_top.append(u + 2*v)

            #Plot predictions with confidence interval
            plt.plot(x, y)
            plt.plot(x, y_bottom, "r")
            plt.plot(x, y_top, "r")
            plt.title(f"Predictions from the Meta Learning Prior with 95% confidence interval : {training_losses_offline[-1]}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig("Prior_Confidence")
            plt.close()


    #Plot training loss vs time
    x = np.linspace(1, len(training_losses_offline), len(training_losses_offline))
    plt.figure(1)
    plt.plot(x, training_losses_offline)
    plt.title("Offline Training Loss vs Time")
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.savefig("Offline_Loss")

    #Plot average variance over time
    plt.figure(2)
    plt.plot(x, average_variances)
    plt.title("Average Offline Variance vs Time")
    plt.xlabel("Training Iterations")
    plt.ylabel("Variance")
    plt.savefig("Average_Offline_Variance")


    #Online Part
    #First want to see what the model returns when predicting over the interval (what the offline posterior parameters return)
    model.start_online()
    x = np.linspace(0, 1, 100)
    y_top = []
    y = []
    y_bottom = []
    for i in range(len(x)):
        pred, sigma = model.online(torch.tensor(x[i:i+1]).float())
        u = pred.detach().numpy()[0][0]
        v = np.sqrt(sigma.detach().numpy()[0][0]) #standard deviation
        y.append(u)
        y_bottom.append(u - 2*v)
        y_top.append(u + 2*v)

    #Plot predictions with confidence interval
    plt.figure(3)
    plt.plot(x, y)
    plt.plot(x, y_bottom, "r")
    plt.plot(x, y_top, "r")
    plt.title("Predictions from the Meta Learning Prior with 95% confidence interval")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("Prior_Confidence")

    #Plot predictions without confidence interval (easier to see)
    plt.figure(4)
    plt.plot(x, y)
    plt.title("Predictions from the Meta Learning Prior")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("Prior_Predictions")


    #Now want to fit to a specific sinusoid with our meta learning prior
    test_freq = 0.5
    #x = np.linspace(0, 1, 100)
    x = np.random.rand(75)
    x_copy = x[:]
    y = np.sin(test_freq * 2 * np.pi*x)
    y_copy = y[:]
    y_pred_top = []
    y_pred = []
    y_pred_bottom = []

    for epoch in range(1):
        plt.figure(epoch+5)
        time.sleep(2)
        for i in range(len(x)):
            pred, sigma = model.online(torch.tensor(x[i:i+1]).float(), update=True, y=torch.tensor(y[i:i+1]).float())
            #u = pred.detach().numpy()[0][0]
            #v = np.sqrt(sigma.detach().numpy()[0][0]) #standard deviation
            #print(v)
            #y_pred.append(u)
            #y_pred_bottom.append(u - 2*v)
            #y_pred_top.append(u + 2*v)

        y_pred = []
        y_pred_bottom = []
        y_pred_top = []
        x = x_copy[:]
        y = y_copy[:]
        for i in range(len(x)):
            pred, sigma = model.online(torch.tensor(x[i:i+1]).float())
            u = pred.detach().numpy()[0][0]
            v = np.sqrt(sigma.detach().numpy()[0][0]) #standard deviation
            #print(v)
            y_pred.append(u)
            y_pred_bottom.append(u - 2*v)
            y_pred_top.append(u + 2*v)

        #Plot predictions
        #plt.figure(5)
        x = x[5:]
        y = y[5:]
        y_pred = y_pred[5:]
        y_pred_top = y_pred_top[5:]
        y_pred_bottom = y_pred_bottom[5:]
        #plt.plot(x, y, "black")
        plt.scatter(x, y_pred)
        plt.scatter(x, y_pred_top, c="r")
        plt.scatter(x, y_pred_bottom, c="r")
        plt.scatter(x, y)
        plt.title("Predictions from the Meta Learning Prior")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("Learned_Sinusoid")
        plt.close()
