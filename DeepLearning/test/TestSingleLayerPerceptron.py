from DeepLearning.NeuralNetwork import NeuralNetwork
from DeepLearning.ActivationFunction import logisticFunction
from numpy import ones,zeros,array
from DeepLearning.TestUtils import assertEquals

class TestSingleLayerPerceptron:
    
    DELTA=1e-8

    def assertNetworkOutput(self, expected, inputData, weights, bias):
        network = NeuralNetwork()
        networkOutput = network.calculateNetworkOutput(inputData, weights, bias, activationFunction=logisticFunction)
        assertEquals(expected.shape[0],networkOutput.shape[0],delta=TestSingleLayerPerceptron.DELTA)
        for i in range(expected.shape[0]):
            assertEquals(expected[i], networkOutput[i], delta=TestSingleLayerPerceptron.DELTA)

    def testZeroWeight(self):
        dimension=1
        bias=0
        inputData=ones(dimension)
        weights=zeros(dimension)
        expected=0.5*ones(dimension)
        self.assertNetworkOutput(expected, inputData, weights, bias)
    
    def testIdentityWeight(self):    
        dimension=1
        bias=0
        inputData=ones(dimension)
        weights=ones(dimension)
        expected=0.731058578 * ones(dimension)
        self.assertNetworkOutput(expected, inputData, weights, bias)
        
    def testNonZeroBias(self):
        dimension=1
        bias=-1
        inputData=ones(dimension)
        weights=ones(dimension)
        expected=0.5*ones(dimension)
        self.assertNetworkOutput(expected, inputData, weights, bias)
        
        
    def testMultipleInputSingleOutputNeurons(self):
        bias=0
        dimension=3
        inputData=0.5*ones(dimension)
        weights=0.5*ones(dimension)
        expected=array([0.679178699])
        print(inputData,weights)
        self.assertNetworkOutput(expected, inputData, weights,bias)
        
    def testSingleInputMultipleOuputNeurons(self):
        bias=-1
        dimension=2
        inputData=array([6.0])
        weights=array([[0.5]*dimension]).T
        expected=array([0.8807970779]*dimension)
        self.assertNetworkOutput(expected, inputData, weights, bias)
    
    def testMultipleInputMultipleOutputNeurons(self):
        bias=-0.5
        dimension=4
        inputData=array([1.75]*dimension)
        weights=array([[0.5]*dimension]*dimension)
        expected=array([0.9525741268]*dimension)
        self.assertNetworkOutput(expected, inputData, weights, bias)
