from numpy import ones, array, copy, linalg, zeros
from DeepLearning.TestUtils import assertEquals
from DeepLearning.UpdateRules import deltaRuleUpdate
from DeepLearning.ActivationFunction import logisticFunctionDerivate, \
    logisticFunction
from DeepLearning.NeuralNetwork import NeuralNetwork


class TestDeltaRuleUpdate:
    
    DELTA = 1e-8

    def assertWeightUpdates(self, expectedUpdates, inputData, targetOutput, learningRate):
        weightUpdates = deltaRuleUpdate(inputData, inputData, targetOutput, learningRate, logisticFunctionDerivate)
        for i in range(expectedUpdates.shape[0]):
            assertEquals(expectedUpdates[i], weightUpdates[i], delta=TestDeltaRuleUpdate.DELTA)

    def testNoChange(self):
        learningRate = 1.0
        dimension = 1
        inputData = ones(dimension)
        targetOutput = copy(inputData)
        expectedUpdates = zeros(inputData.shape)
        self.assertWeightUpdates(expectedUpdates, inputData, targetOutput, learningRate)
    
    def testSingleChange(self):
        learningRate = 1.0
        dimension = 1
        inputData = 0.5 * ones(dimension)
        targetOutput = 0.6 * ones(dimension)
        expectedUpdates = 0.0117501856 * ones(inputData.shape)
        self.assertWeightUpdates(expectedUpdates, inputData, targetOutput, learningRate)
        
    def testLearningRate(self):
        dimension = 1
        inputData = 0.5 * ones(dimension)
        targetOutput = 0.6 * ones(dimension)
        learningRate = 0.0
        expectedUpdates = learningRate * 0.0117501856 * ones(inputData.shape)
        self.assertWeightUpdates(expectedUpdates, inputData, targetOutput, learningRate)
        learningRate = -1.0
        expectedUpdates = learningRate * 0.0117501856 * ones(inputData.shape)
        self.assertWeightUpdates(expectedUpdates, inputData, targetOutput, learningRate)
        learningRate = 2.0
        expectedUpdates = learningRate * 0.0117501856 * ones(inputData.shape)
        self.assertWeightUpdates(expectedUpdates, inputData, targetOutput, learningRate)
        
    def testMultipleOutputNeurons(self):
        dimension = 3
        inputData = -0.2 * ones(dimension)
        targetOutput = 0.5 * ones(dimension)
        learningRate = 0.03
        expectedUpdates = -0.001039569605 * ones(inputData.shape)
        self.assertWeightUpdates(expectedUpdates, inputData, targetOutput, learningRate)
