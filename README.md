# Feed Forward Neural Network from scratch
* This reposistory contains the code and mathematical calculations involved in building a neural network from scratch by me.
## Files in the repository
* Following are the details of the files present in the repository.
* **Modular_Feed_Forward_Neural_Network_From_Scratch.ipynb** is the jupyter notebook having the main code.
  
* **modular_feed_forward_neural_network_from_scratch.py** is the python3 file for the same code.
  
* **Involved_Mathematical_Calculations.pdf** is the pdf file having all the mathematical calculations involved in the Neural Network.
  
* **Backpropagation.pdf** is the mathematically derived **pseudocode** for backpropagation algorithm used in the Neural Network with **softmax** as the output function.

## Architecture of the Feed Forward Neural Network
* Following image explains the architecture of the neural network.
[![feedfwd-nn-example-cs7015-page-0001.jpg](https://i.postimg.cc/mrBPYThb/feedfwd-nn-example-cs7015-page-0001.jpg)](https://postimg.cc/rDZycXSb)
* The output function used here is **Softmax** function.
[![Softmax.png](https://i.postimg.cc/C1dL750V/Softmax.png)](https://postimg.cc/9R5hFWzx)
```python
import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Subtract max(z) for numerical stability
    return exp_z / exp_z.sum(axis=0)

# Example usage
logits = np.array([2.0, 1.0, 0.1])
softmax_probs = softmax(logits)

print("Logits:", logits)
print("Softmax Probabilities:", softmax_probs)
```
## Modularity of code
* The code of **Neural Network** is modular and user friendly.
* We can specify the required **activation function**,**input nodes**, **output classes**.
* Currently the model is ready for **classification** tasks, but just by changing the **output function**, we can change it do **regression** tasks as well.
* For **regression tasks**, we can use a **linear output function**.

## Statistically optimised weights and biases initialisation
[![statistically-optimised-initialisation.jpg](https://i.postimg.cc/Pxm8d56Z/statistically-optimised-initialisation.jpg)](https://postimg.cc/LY8sT2F6)
```python
W = np.random.randn(in_nodes,out_nodes)/sqrt(in_nodes)
```
* **in_nodes** is the number of input nodes connected to the neuron and **out_nodes** is the number of output nodes the neuron is connected with.


