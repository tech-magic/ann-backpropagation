package neurons;

/**
 * This represents a single layer in the neural network
 * A neural network is an array of layers;
 * <pre>
 * 1. the input layer
 * 2. the output layer
 * 3. hidden layers
 * </pre>
 * @author wimal perera (09/10008)
 *
 */
public class Layer {
	
	/**
	 * Our layer has an array of perceptrons
	 */
	private Perceptron[] perceptrons;
	
	/**
	 * The index of the layer with respect to the
	 * neural network
	 */
	private int index;
	
	private int perceptronCount;
	
	public Layer(int index, int perceptronCount) {
		this.index = index;
		this.perceptrons = new Perceptron[perceptronCount];
		this.perceptronCount = perceptronCount;
	}
	
	/**
	 * Method used to retrieve a perceptron from this layer
	 * @param index
	 * @return
	 */
	public Perceptron getPerceptron(int index) {
		if(index < perceptronCount)
			return this.perceptrons[index];
		else
			throw new RuntimeException(index + " is out of Range, perceptron count for layer " + this.index + " is : " + perceptronCount);
	}
	
	/**
	 * Method used to add a perceptron to this layer
	 * @param index
	 * @param perceptron
	 */
	public void addPerceptron(int index, Perceptron perceptron) {
		if(index < perceptronCount)
			this.perceptrons[index] = perceptron;
		else
			throw new RuntimeException(index + " is out of Range, perceptron count for layer " + this.index + " is : " + perceptronCount);
	}
	
	/**
	 * Method used to obtain the number of perceptrons (size)
	 * in this layer.
	 * @return
	 */
	public int getPerceptronCount() {
		return this.perceptronCount;
	}
}
