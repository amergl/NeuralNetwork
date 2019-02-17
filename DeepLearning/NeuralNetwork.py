from numpy import dot, array, zeros

class NeuralNetwork:
    
    def calculateNeuronInput(self, inputData, weights, bias):
        return dot(weights,inputData)+bias
    
    def calculateNetworkOutput(self,inputData,weights,bias,activationFunction):
        nOutputNeurons=1
        if len(weights.shape) > 1:
            nOutputNeurons=weights.shape[0]
        else:
            weights=array([weights])
        neuronOutput=zeros(nOutputNeurons)
        for i in range(nOutputNeurons):
            neuronInput=self.calculateNeuronInput(inputData, weights[i], bias)
            neuronOutput[i]=activationFunction(neuronInput)
        return array(neuronOutput,dtype=float)
