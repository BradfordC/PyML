import Network

net = Network.Network(5, 5, [5, 6, 7])
print(net.GetLayerSizes())
print(net.GetLayerSizes()[1:-1])