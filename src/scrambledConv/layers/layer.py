class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient):
        # TODO: return gradients
        pass
    
    def learning(self, weights_grad, bias_grad):
        # TODO: apply gradients
        pass
