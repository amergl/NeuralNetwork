from numpy import dot
class NeuralNetwork:
    
    def calculateNeuronInput(self, inputData, weights, bias):
        return dot(weights,inputData)+bias
    
    
