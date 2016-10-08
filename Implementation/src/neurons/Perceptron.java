package neurons;

import maths.ActivationFunction;
import maths.MathUtils;

/**
 * This class represents the
 * perceptrons found in our neural network.
 * Note that for implementation purposes
 * our input terminals (i.e. input layer) are also 
 * considered as a special kind of perceptrons 
 * (please refer to InputTerminal class).
 * Further the perceptrons in output layer are
 * a bit different than others (please refer to the
 * OutputPerceptron class). 
 * @author wimal perera (09/10008)
 *
 */
public class Perceptron {
	
	/**
	 * Inputs and their corresponding
	 * weights
	 */
	private boolean[] inputs;
	private float[] weights;
	private int inputSize;
	
	/**
	 * When we take the output we store
	 * it in the perceptron itself since we
	 * need it later when adjusting weights
	 */
	private boolean storedOutput;
	private float storedWeightedSum;
	
	/**
	 * This is the delta value which we calculate during
	 * the backward pass of the algorithm.
	 */
	private float delta;
	
	/**
	 * Threshold, bias and activation function
	 */
	private float threshold;
	private float bias;
	private ActivationFunction activationFunction = ActivationFunction.LINEAR;
		
	public Perceptron(int inputSize) {
		this(inputSize, 0.0f, ActivationFunction.LINEAR, 0.0f);
	}
	
	public Perceptron(int inputSize, 
			ActivationFunction activationFunction,
			float threshold) {
		this(inputSize, 0.0f, activationFunction, threshold);
	}
	
	public Perceptron(int inputSize, 
			float bias, 
			ActivationFunction activationFunction, 
			float threshold) {
		
		this.inputSize = inputSize;
		this.inputs = new boolean[inputSize];
		this.weights = new float[inputSize];
		
		this.bias = bias;
		this.threshold = threshold;
		this.activationFunction = activationFunction;
	}
	
	public void setWeight(int index, float weight) {
		if(index < inputSize)
			this.weights[index] = weight;
		else
			throw new RuntimeException(index + " is out of Input Range, input size is : " + inputSize);
	}
	
	public void setInput(int index, boolean input) {
		if(index < inputSize)
			this.inputs[index] = input;
		else
			throw new RuntimeException(index + " is out of Input Range, input size is : " + inputSize);
	}
	
	public boolean getInput(int index) {
		if(index < inputSize)
			return this.inputs[index];
		else
			throw new RuntimeException(index + " is out of Input Range, input size is : " + inputSize);
	}
	
	public float getWeight(int index) {
		if(index < inputSize)
			return this.weights[index];
		else
			throw new RuntimeException(index + " is out of Input Range, input size is : " + inputSize);
	}
	
	public float getBias() {
		return this.bias;
	}
	
	public void calculateOutput() {
		
		//obtain the weighted sum
		float sum = 0.0f;
		for(int i = 0; i < inputSize; i++) {
			sum += weights[i] * MathUtils.booleanToFloat(inputs[i]);
		}
		
		//we store the weighted sum in the perceptron itself
		this.storedWeightedSum = sum;
		
		//we have the only sigmoid activation function for now
		//except for the input terminals
		//we store the output in the perceptron itself
		if(activationFunction == ActivationFunction.SIGMOID) {
			float sigmoid = MathUtils.sigmoid(sum);
			if(sigmoid < threshold) {
				this.storedOutput = false;
			}
			else {
				this.storedOutput = true;
			}
		} 
	}
	
	/**
	 * This method is required when re-adjusting weights 
	 * @return
	 */
	public float getDifferentialOutput() {
		if(this.activationFunction == ActivationFunction.SIGMOID) {
			return MathUtils.diffSigmoid(this.storedWeightedSum);
		}
		else 
			return 0;
	}

	public void setDelta(float delta) {
		this.delta = delta;
	}

	public float getDelta() {
		return delta;
	}
	
	protected void setStoredOutput(boolean storedOutput) {
		this.storedOutput = storedOutput;
	}
	
	public boolean getStoredOutput() {
		return this.storedOutput;
	}
	
	public int getInputSize() {
		return this.inputSize;
	}
}
