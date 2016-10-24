import GetInput
import Network
import numpy as np

#Data = GetInput.LoadAllCategories("D:/Chris/Documents/School/Machine Learning/Project/4Categories", True)

net = Network.Network(3, 5, [3,4,5])
output = net.Feedforwad(np.array([2,3,4]))
print(net.GetError(np.array([1, 1, 1, 1, 1])))