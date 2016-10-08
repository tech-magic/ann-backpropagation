package maths;

/**
 * This eneumeration is used by the perceptron class
 * to determine the activation function it should
 * use when calculating the output.
 * 
 * Note that LINEAR is used only for input terminals
 * where you get the input as the output as it is.
 * @author wimal perera (09/10008)
 *
 */
public enum ActivationFunction {
	LINEAR,
	SIGMOID;
}
