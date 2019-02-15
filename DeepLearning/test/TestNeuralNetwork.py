from DeepLearning.NeuralNetwork import NeuralNetwork
class TestNeuralNetwork:
    
    def testConstructor(self):
        network=NeuralNetwork()
        assert network is not None
