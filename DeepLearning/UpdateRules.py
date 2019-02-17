from numpy import array, zeros


def deltaRuleUpdate(inputData, actualOutput, targetOutput, learningRate, activateFunctionDerivate):
    deltaOutput = targetOutput - actualOutput
    derivateVector = zeros(inputData.shape)
    for i in range(inputData.shape[0]): 
        derivateVector[i] = activateFunctionDerivate(inputData[i])
    weightUpdates = zeros(inputData.shape)
    for i in range(inputData.shape[0]):
        weightUpdates[i] = learningRate * deltaOutput[i] * derivateVector[i] * inputData[i]
    return weightUpdates
