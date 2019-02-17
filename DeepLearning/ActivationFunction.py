from numpy import exp
def logisticFunction(xValue):
    return 1.0/(1+exp(-xValue))

def logisticFunctionDerivate(xValue):
    return logisticFunction(xValue)*(1.0-logisticFunction(xValue))