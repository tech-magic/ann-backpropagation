package neurons;

import maths.ActivationFunction;

/**
 * These perceptrons are used to represent perceptrons
 * in the output layer.
 * 
 * Output perceptrons are different from others as they
 * have a desired output.
 * 
 * @author wimal perera (09/10008)
 *
 */
public class OutputPerceptron extends Perceptron {
	
	private boolean desiredOutput;
	
	public OutputPerceptron(int inputSize, 
			ActivationFunction activationFunction,
			float threshold) {
		super(inputSize, 0.0f, activationFunction, threshold);
	}

	public void setDesiredOutput(boolean desiredOutput) {
		this.desiredOutput = desiredOutput;
	}

	public boolean getDesiredOutput() {
		return desiredOutput;
	}
	
	

}
