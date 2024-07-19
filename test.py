import numpy as np

def activation(x,type):

  if type == 'sigmoid':
    return 1/(1 + np.exp(-x))

  elif type == 'tanh':
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    
  elif type == 'ReLU':
    return np.maximum(0,x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)




class Neural_Network():


  #INITIALISING FUNCTION
  def __init__(self,input_nodes,layers,output_nodes):
    self.input_nodes = input_nodes  #dimension of the input vector
    self.layers = layers   #number of layers that we want in netork
    self.output_nodes = output_nodes  #number of output classes

    #convert to small_named variables for comfort
    self.n = self.input_nodes
    self.L = self.layers
    self.k = self.output_nodes

    self.W = []  # W will be a list of matrices of size nxn
    self.B = []  # B will be a list of matrices of size 1xn

    #weights & biases initialisation for layers 1 to L-1
    for i in range(0, self.L-1):
        W_i = np.random.rand(self.n, self.n)/np.sqrt(self.n) # creates nxn matrix
        B_i = np.random.rand(1, self.n)  # creates 1xn matrix
        self.W.append(W_i)
        self.B.append(B_i)

    #weights & biases initialisation for last layer
    W_i = np.random.rand(self.n,self.k)/np.sqrt(self.n) # creates nxk matrix
    B_i = np.random.rand(1,self.k) # creates 1xk matrix
    self.W.append(W_i)
    self.B.append(B_i)

  #FORWARD PROPAGATION THROUGH THE LAYERS
  def forward(self,X):
    Y_hat = []
    # Initialize lists A and H dynamically based on the value of L
    A = [[] for _ in range(self.layers)]
    H = [[] for _ in range(self.layers)]

    for x in X:
      

      #input layer
      a0 = np.dot(self.W[0],x.T)  + self.B[0].T   #a1 = W1.x + b1
      A[0].append(a0)
      h0 = activation(a0,type='sigmoid')  #h1 = activation(a1)
      H[0].append(h0)

      #middle layers
      for i in range(1,self.L-1):
          # Access the value in H[i-1] corresponding to the current x
          h_prev = H[i - 1][-1]  # Get the last element added to H[i-1]
          a = np.dot(self.W[i], h_prev) + self.B[i].T  # a = Wi.h(i-1) + Bi
          A[i].append(a)
          h = activation(a, type='sigmoid')
          H[i].append(h)

      #output layer
      a_L = np.dot(self.W[self.L-1].T,H[self.L-2][-1]) + self.B[self.L-1].T
      A[self.L-1].append(a_L)
      y_hat = softmax(a_L)
      Y_hat.append(y_hat)

    return Y_hat, A, H    #each y_hat is a kx1 matrix
  
  
  
  #BACKPROPAGATION THROUGH THE LAYERS
  def backward(self,X,Y,lr,max_iter):
    
    for i in range(max_iter):
      dW = []
      dB = []
      dA = [[] for _ in range(self.layers)]
      dH = [[] for _ in range(self.layers)]

      for i in range(0, self.L-1):
        dW_i = np.zeros((self.n, self.n))  # creates nxn matrix
        dB_i = np.zeros((1, self.n))  # creates 1xn matrix
        dW.append(dW_i)
        dB.append(dB_i)

      dW_i = np.zeros((self.n, self.k))  # creates nxk matrix for last layer
      dB_i = np.zeros((1, self.k))  # creates 1xk matrix for last layer
      dW.append(dW_i)
      dB.append(dB_i)

      Y_hat, A, H  = self.forward(X)

      for x, y, y_hat in zip(X, Y, Y_hat):
        #backpropagate through output layer
        grad_A_L_minus_1 = (y_hat - y)
        dA[self.L-1].append(grad_A_L_minus_1)

        #backpropagate through middle layers
        for j in range(self.L-1, 0, -1):
          dw = np.dot(grad_A_L_minus_1, H[j-1][-1].T)
          dW[j] += dw
          dB[j] += grad_A_L_minus_1
          grad_H_j_minus_1 = np.dot(self.W[j].T, grad_A_L_minus_1)
          dH[j-1].append(grad_H_j_minus_1)
          grad_A_j_minus_1 = grad_H_j_minus_1 * activation(A[j-1][-1], type='sigmoid', derivative=True)
          dA[j-1].append(grad_A_j_minus_1)

        #backpropagate through input layer
        dw = np.dot(grad_A_j_minus_1, x.T)
        dW[0] += dw
        dB[0] += grad_A_j_minus_1

      #update the weights and biases
      for j in range(self.L):
        self.W[j] -= lr * dW[j]
        self.B[j] -= lr * dB[j]
        
    return self.W,self.B
        
    