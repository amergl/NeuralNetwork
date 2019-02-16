from DeepLearning import ActivationFunction
from DeepLearning.TestUtils import assertEquals
from matplotlib.pyplot import plot,show
from DeepLearning.NeuralNetwork import NeuralNetwork
from numpy import zeros,ones

class TestEvaluation:

    DELTA=1e-32

    def testLogisticFunction(self):
        logisticFunction=ActivationFunction.logisticFunction
        negativeInput=-1e2
        value=logisticFunction(negativeInput)
        assertEquals(0.0,value,delta=TestEvaluation.DELTA)
        zeroInput=0.0
        value = logisticFunction(zeroInput)
        assertEquals(0.5,value,delta=TestEvaluation.DELTA)
        largeInput=1e2
        value = logisticFunction(largeInput)
        assertEquals(1.0, value, delta=TestEvaluation.DELTA)
        
    def testWeightEvaluation(self):
        dimension=5
        inputData=zeros(dimension)
        weights=zeros(dimension)
        self.assertNeuronInput(inputData,weights,0.0,expected=0.0)
        self.assertNeuronInput(inputData,weights,1.0,expected=1.0)
        inputData[0]=1
        self.assertNeuronInput(inputData,weights,1.0,expected=1.0)
        inputData[0]=1
        weights[0]=1
        self.assertNeuronInput(inputData,weights,0.0,expected=inputData[0])
        inputData[0]=1
        weights[0]=0.5
        self.assertNeuronInput(inputData, weights, 0.0, expected=weights[0])
        inputData=ones(dimension)
        weights=ones(dimension)
        self.assertNeuronInput(inputData, weights, 0.0, expected=dimension)
        inputData=ones(dimension)
        weights=ones(dimension)
        expected=dimension+1
        self.assertNeuronInput(inputData, weights, 1.0, expected=expected)
             
    def assertNeuronInput(self,inputData,weights,bias,expected=0):
        network=NeuralNetwork()
        neuronInput=network.calculateNeuronInput(inputData,weights,bias)
        assertEquals(expected,neuronInput,delta=TestEvaluation.DELTA)
        
    def xtestPlot(self):
        logisticFunction=ActivationFunction.logisticFunction
        x=range(-10,10)
        v=[logisticFunction(y) for y in x]
        plot(x,v)
        show()
            