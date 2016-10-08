package neurons;

import maths.ActivationFunction;

/**
 * This class is used to represent the input terminals
 * for our neural network.
 * 
 * Note that for convenience in implementation, I've
 * used the input terminals, as a special form of
 * perceptrons where you have one input terminal and
 * one output terminal and you always get the input 
 * value as the output
 * 
 * @author wimal perera (09/10008)
 *
 */
public class InputTerminal extends Perceptron {
	
	public InputTerminal() {
		super(1, 0.0f, ActivationFunction.LINEAR, 0.0f);
		super.setWeight(0, 1.0f);
	}
	
	/**
	 * We set up the input here.
	 * The stored output is the same as the input.
	 * @param input
	 */
	public void setInput(boolean input) {
		super.setStoredOutput(input);
		super.setInput(0, input);
	}
	
	@Override
	public void calculateOutput() {
		//we don't use this method for input terminals
	}
}
