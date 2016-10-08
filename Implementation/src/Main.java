import neurons.NeuralNetwork;

/**
 * This is the test class indicating how to
 * use our neural network for training
 * using the backpropagation algorithm
 * @author wimal perera (09/10008)
 *
 */
public class Main {
	
	public static void main(String args[]) throws Exception {
		
		// this is our main test-bed
		Main main = new Main();
		main.testForXOR();
		main.testForMyNetwork();
	}
	
	/**
	 * This method is used to demonstrate that my neural network class
	 * works for simple XOR network with a single hidden layer
	 */
	public void testForXOR() {
		
		//parameters for creating the neural network
		int inputTerminals = 2;
		int outputTerminals = 1;
		float learningRate = 0.1f;
		float commonThreshold = 0.5f;
		int[] hiddenLayerInfo = new int[] {2};
		
		// the neural network for XOR has 2 input terminals 
		// in the input layer, 1 output terminal at the output
		// layer and a single hidden layer with 2 perceptrons
		// note that we use a common threshold of 0.5 for
		// each perceptron and the learning rate is 0.1.
		// further for each input in each perceptron weights
		// are randomly assigned as a value between -1.0f to
		// 1.0f when the neural network is initially created.
		NeuralNetwork neuralNet = new NeuralNetwork(
				inputTerminals, outputTerminals, 
				learningRate, commonThreshold, hiddenLayerInfo);
		
		//example training data for the neural network
		boolean[][] inputVectors = new boolean[][] {
				{false, false},
				{false, true},
				{true, false},
				{true, true}
		};
		boolean[][] desiredOutputVectors = new boolean[][] {
				{false},
				{true},
				{true},
				{false}
		};
		int iterations = 50;
		String outputFile = "xor_output.txt";
		
		//we train our neural network for 50 iterations
		//the resulting weights of each perceptron in
		//each layer is appended to the output file
		//xor_output.txt
		neuralNet.trainNeuralNet(
				outputFile, iterations, 
				inputVectors, desiredOutputVectors);
	}
	
	/**
	 * This method is used to demonstrate that my neural network class
	 * works for the neural network which was requested for me to
	 * implement during the assignment.
	 */
	public void testForMyNetwork() {
		
		//data for creating the neural network
		int inputTerminals = 6;
		int outputTerminals = 7;
		float learningRate = 0.1f;
		float commonThreshold = 0.5f;
		int[] hiddenLayerInfo = new int[] {8, 8, 8, 8, 8, 8};
		
		// the neural network in the assignment has 6 input terminals 
		// in the input layer, 7 output terminals at the output
		// layer and 6 hidden layers (I'm using 8 perceptrons in each)
		// note that we use a common threshold of 0.5 for
		// each perceptron and the learning rate is 0.1.
		// further for each input in each perceptron weights
		// are randomly assigned as a value between -1.0f to
		// 1.0f when the neural network is initially created.
		NeuralNetwork neuralNet = new NeuralNetwork(
				inputTerminals, outputTerminals, 
				learningRate, commonThreshold, hiddenLayerInfo);
		
		//example training data for the neural network
		boolean[][] inputVectors = new boolean[][] {
				{false, false, false, false, false, true},
				{false, true,  true,  false, true,  false},
				{true,  false, true,  true,  false, true},
				{true,  true,  false, false, true,  true},
				{true,  true,  false, true,  true,  true},
				{true,  false, false, true,  false, true},
				{true,  false, false, true,  false, false}
		};
		boolean[][] desiredOutputVectors = new boolean[][] {
				{false, false, false, false, false, true,  false},
				{true,  false, false, true,  false, false, true},
				{true,  false, false, true,  true,  false, true},
				{true,  false, false, false, true,  false, true},
				{true,  true,  true,  false, true,  true,  false},
				{true,  false, true,  true,  false, true,  false},
				{false, false, true,  false, true,  false, false}
		};
		
		int iterations = 50;
		String outputFile = "mynet_output1.txt";
		
		//we train our neural network for 50 iterations
		//the resulting weights of each perceptron in
		//each layer is appended to the output file
		//mynet_output1.txt
		neuralNet.trainNeuralNet(
				outputFile, iterations, 
				inputVectors, desiredOutputVectors);
	}
}
