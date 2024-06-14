#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import random


# In[2]:


def synthetic_LRdata(w, sigX, n, muX = None):
    # synthesize Linear regression data with given parameter (w,b), 
    # Variance of X and \epsilon, number of samples n.
    # Output the torch tensor data X, y
    if w.dtype != torch.float64:
        w = w.to(torch.float64)
    #print(w.dtype)
    if muX == None:
        muX = torch.zeros(len(w))

    X = np.random.multivariate_normal(mean=muX, cov = sigX*np.identity(len(w)), size=n)
    for i in range(n):
        X[i, 0] = 1.
    X = torch.tensor(X)
    #print(X)
    #print(X,w)
    y = torch.matmul(X, w) 
    #y += torch.normal(0, sige, y.shape)
    return X, y.reshape((-1,1))


# In[3]:


# Generate the adjacency matrix
def ring(n):
    A = np.zeros((n, n))
    for i in range(n-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    A[0, n-1] = 1
    A[n-1, 0] = 1
    return A

def erdos_renyi(n, p):
    # Generate an Erdős-Rényi graph with parameters (n, p)
    er_graph = nx.erdos_renyi_graph(n, p)
    
    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_matrix(er_graph)
    
    return adjacency_matrix

def watts_strogatz(n, k, p):
    # Generate a Watts-Strogatz graph with parameters (n, k, p)
    ws_graph = nx.watts_strogatz_graph(n, k, p)
    
    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_matrix(ws_graph)
    
    return adjacency_matrix

def grid(rows, cols):
    total_nodes = rows * cols
    adjacency_matrix = np.zeros((total_nodes, total_nodes), dtype=int)
    
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            
            if i > 0:
                adjacency_matrix[node, node - cols] = 1  # Upper neighbor
            if i < rows - 1:
                adjacency_matrix[node, node + cols] = 1  # Lower neighbor
            if j > 0:
                adjacency_matrix[node, node - 1] = 1      # Left neighbor
            if j < cols - 1:
                adjacency_matrix[node, node + 1] = 1      # Right neighbor
                
    return adjacency_matrix

def twoblock(n1, n2, p_high, p_low):
    n = n1 + n2  # Total number of vertices
    
    # Create an empty adjacency matrix
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    # Create the first community
    for i in range(n1):
        for j in range(i + 1, n1):
            if np.random.rand() < p_high:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    
    # Create the second community
    for i in range(n1, n1 + n2):
        for j in range(i + 1, n1 + n2):
            if np.random.rand() < p_high:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    
    # Connect between communities
    for i in range(n1):
        for j in range(n1, n):
            if np.random.rand() < p_low:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
                    
    return adjacency_matrix


# In[4]:


# Other Utils
def linreg(X, w):
    return(torch.matmul(X, w))
    
def mseloss(y, y_hat, l):
    return((y_hat - y.reshape(y_hat.shape))**2/(2*l))

#def sgd(params, lr, weight = 1):  
#    with torch.no_grad():
#        param -= lr * weight * param.grad
#        param.grad.zero_()
        
def MH(G):
    # Uniform via MH
    n = G.shape[0]
    deg = np.zeros(n)
    for k in range(n):
        deg[k] = G[k, :].sum()
    Pdeg=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if G[i,j] == 1 and i != j:
                Pdeg[i,j] = np.minimum(1., (deg[i])/(deg[j]))/deg[i]
        Pdeg[i, i] = np.maximum(0, 1 - np.sum(Pdeg[i, :]))
    return Pdeg

def MHI(G, X):
    # Importance Sampling via MH
    n = X.shape[0]
    deg = [np.sum(G[i,:]) for i in range(n)]
    #print(deg)
    Lip = [np.linalg.norm(X[i,:]) for i in range(n)]
    Lip = np.array(Lip)**2
    plip=Lip/np.sum(Lip)
    Plip=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if G[i,j] == 1 and i != j:
                Plip[i,j] = np.minimum(1., deg[i]*Lip[j]/(deg[j]*Lip[i]))/deg[i]
        Plip[i, i] = np.maximum(0,1 - np.sum(Plip[i, :]))
    return Plip


# In[5]:


# Jump matrix
def GtoP(G):
    G = np.array(G, dtype = float)
    for i in range(G.shape[0]):
        w = np.sum(G[i, :])
        #print(w)
        G[i, :] = G[i, :]/w
    return G

def truncateGeom(p, i, r):
    pd = p * (1 - p) ** (i) / (1 - ( 1 - p ) ** (r))
    return pd

def Kth_neighbor(G, k, p):
    GK = [G]
    P = GtoP(GK[0]) * truncateGeom(p, 0, k)
    for i in range(k):
        GK.append(GK[i]@G)
    for i in range(k-1):
        P += GtoP(GK[i+1]) * truncateGeom(p, i+1, k)
    return P

def TruncGeom(pd, r):
    ber = np.random.uniform()
    prob_dist = pd/(1-(1-pd)**r)
    #print(prob_dist)
    d = 1
    while ber > prob_dist:
        d += 1
        prob_dist += pd*(1-pd)**(d-1)/(1-(1-pd)**r)
        #print(prob_dist)
    return d


# In[6]:


# main class
class RWnode:
    def __init__(self, X, y, W1, W2, init_model, pi, index ):
        self.index = index
        self.X = X
        self.y = y
        self.W = {'Uniform':W1, 'Importance':W2}
        self.neighbors = []
        self.model = init_model
        self.loss = []
        self.pi = pi
        self.index = index
        
        
        #print(self.W)
        
    def localSGD(self, gamma, Xtrain, ytrain, weight = 1):
        l = mseloss(self.y, linreg(self.X, self.model), l = 1)
        l.backward()
        model =  self.model - gamma * self.model.grad / weight
        self.model = model.clone().detach().requires_grad_(True)
        '''with torch.no_grad():  
            n = len(ytrain)
            train_l = mseloss(ytrain, linreg(Xtrain, self.model), l = n)
            #print(train_l.mean())
            self.loss.append(train_l.mean())'''

class RW_model:
    def __init__(self, G, X, y, k = None):
        if k == None:
            print('No jump.')
            #return
        else:
            print(f'Levy jump {k}')
            self.Kth_P = self.__Kth_neighbor(G, k)
        self.n = X.shape[0]
        Lip = [np.linalg.norm(X[i,:]) for i in range(self.n)]
        Lip = np.array(Lip)**2
        plip=Lip/np.sum(Lip)
        self.lip = torch.tensor(plip)
        self.__initialization(G, X, y,  plip)
        self.X = X
        self.y = y
        self.loss = []
        #self.loss_communication = []
        self.current = self.nodes[0]
        self.updates = [0]
        self.communication = [0]
        
    def __Kth_neighbor(self, G, k):
        GK = [G]
        P = [GtoP(GK[0])] 
        for i in range(k-1):
            GK.append(GK[i]@G)
        for i in range(k-1):
            P.append(GtoP(GK[i+1]))
        return P
    
    def __GtoP(self, G):
        G = np.array(G, dtype = float)
        for i in range(G.shape[0]):
            w = np.sum(G[i, :])
        #print(w)
            G[i, :] = G[i, :]/w
        return G
        
        
    def __initialization(self, G, X, y, plip):
        (n, p) = X.shape
        W1 = MH(G)
        W2 = MHI(G, X)
        initial_model = torch.tensor(np.random.normal(0, 1, p))
        self.nodes = [RWnode(X[i, :], y[i], W1[i, :], W2[i, :],  initial_model.detach().clone().requires_grad_(True),  plip[i], i) for i in range(n)]
        for node_ind in range(n):
            for neighbor_ind in range(n):
                if G[node_ind, neighbor_ind] == 1:
                    Node = self.nodes[node_ind]
                    Node.neighbors.append(self.nodes[neighbor_ind])
                    
    def re_init(self, model):
        for node in self.nodes:
            node.model = model
                    
    def optimize(self, gamma = 0.01, jump = None, mh = 'Importance'):
        node = self.current
        if mh == 'Importance':
            node.localSGD(gamma, self.X, self.y, weight = node.pi * self.n)
        else:
            node.localSGD(gamma, self.X, self.y)
        self.loss.append(self.cal_loss())
        if jump == None:
            nextnode = np.random.choice(self.nodes, size = 1, p = node.W[mh])[0]
            if nextnode != self.current:
            #print('2')
            #    self.communication.append(self.communication[-1] + 1)
                self.commute = True
            else:
            #print('1')
                #self.communication.append(self.communication[-1])
                self.commute = False
        else:
            nextnode = np.random.choice(self.nodes, size = 1, p = self.Kth_P[int(jump)-1][node.index])[0]
            #for _ in range(jump):
            #    self.communication.append(self.communication[-1] + 1)
            self.commute = True
        
        
        self.updates.append(self.updates[-1] + 1)
        
        # pass the model to the next node
        nextnode.model = node.model.clone().detach().requires_grad_(True)
        self.current = nextnode
        #optimize
    
    def cal_loss(self):
        with torch.no_grad():  
            n = len(self.y)
            train_l = mseloss(self.y, linreg(self.X, self.current.model), l = n)
            #print(train_l.mean())
        return train_l.mean()


# In[7]:


def computeloss_mh(Test1, numpath, maxstep, cut, gamma, init, mh = 'Importance'):
    Loss_communication = []
    Loss_update = []
    for _ in range(numpath):
        initialmodel = init.clone().detach().requires_grad_(True)
        Test1.re_init(initialmodel)
        Test1.current = Test1.nodes[-1]
        loss_communication = [0]
        loss_update = [0]
        for i in range(maxstep):
            Test1.optimize(gamma, jump = None, mh = mh)
            current_loss = Test1.cal_loss()
            loss_update.append(current_loss)
            if Test1.commute:
                loss_communication.append(current_loss)
            else:
                loss_communication[-1] = current_loss
        Loss_update.append(loss_update)
        Loss_communication.append(loss_communication)
    #print(len(Loss_communication[0]), len(Loss_communication[1]))
    #Loss_update = np.array(Loss_update)
    #print(Loss_update.shape)
    #Loss_communication = np.array(Loss_communication)
    #print(Loss_communication.shape)
    return Loss_update,  Loss_communication 


# In[8]:


# Return the loss per iteration
def computeloss_jump(Test1, numpath, maxstep, cut, pj, pd, r, gamma, init):
    #initialmodel = Test1.nodes[0].model.clone().detach().requires_grad_(True)
    Loss_communication = []
    Loss_update = []
    for _ in range(numpath):
        initialmodel = init.clone().detach().requires_grad_(True)
        Test1.re_init(initialmodel)
        Test1.current = Test1.nodes[-1]
        loss_communication = [0]
        loss_update = [0]
        for i in range(maxstep):
            if np.random.binomial(1, pj):
                Test1.optimize(gamma)
                current_loss = Test1.cal_loss()
                loss_update.append(current_loss)
                if Test1.commute:
                    loss_communication.append(current_loss)
                else:
                    loss_communication[-1] = current_loss
            else:
                d = TruncGeom(pd, r)
                Test1.optimize(gamma, jump = d)
                current_loss = Test1.cal_loss()
                loss_update.append(current_loss)
                for _ in range(d):
                    loss_communication.append(current_loss)
        Loss_update.append(loss_update[:cut])
        Loss_communication.append(loss_communication[:cut])
    print(len(Loss_communication[0]), len(Loss_communication[1]))
    Loss_update = np.array(Loss_update) 
    Loss_communication = np.array(Loss_communication) 
    return np.mean(Loss_update ,axis=0), np.mean(Loss_communication ,axis=0)


# In[ ]:


lc_is, lu_is = computeloss_mh(RW2, 50, 200000, 30000, init = initialmodel, gamma=0.01 )

