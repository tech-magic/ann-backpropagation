package neurons;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import maths.ActivationFunction;
import maths.MathUtils;

/**
 * This is the core class representing the feedforward
 * neural network and the backpropagation algorithm.
 * @author wimal perera (09/10008)
 *
 */
public class NeuralNetwork {
	
	/**
	 * The layers in this neural network;
	 * <pre>
	 * 1. Input Layer
	 * 2. Output Layer
	 * 3. Hidden Layers
	 * </pre>
	 */
	private Layer[] layers;
	
	/**
	 * Basic settings to set up the
	 * neural network
	 */
	private int inputTerminalCount;
	private int outputTerminalCount;
	private int hiddenLayerCount;
	private int[] hiddenLayerSizes;
	
	/**
	 * Basic settings helpful when training
	 * the neural network
	 */
	private float learningRate;
	private float commonThreshold;
	
	/**
	 * This is the constructor to create a neural
	 * network as we wish.
	 * @param inputTerminalCount
	 * @param outputTerminalCount
	 * @param learningRate
	 * @param commonThreshold
	 * @param hiddenLayerSizes
	 */
	public NeuralNetwork( 
			int inputTerminalCount, 
			int outputTerminalCount,
			float learningRate,
			float commonThreshold,
			int[] hiddenLayerSizes) {
		
		this.hiddenLayerSizes = hiddenLayerSizes;
		this.hiddenLayerCount = hiddenLayerSizes.length;
		
		//the total layers for this network is
		// hidden layer count + input layer + output layer
		this.layers = new Layer[this.hiddenLayerCount + 2];
		
		this.inputTerminalCount = inputTerminalCount;
		this.outputTerminalCount = outputTerminalCount;
		
		this.learningRate = learningRate;
		
		// we set up our initial neural network
		// using the below 3 methods.
		this.buildInputLayer();
		this.buildHiddenLayers();
		this.buildOutputLayer();
		
		this.commonThreshold = commonThreshold;
	}
	
	/**
	 * This is how we build the input layer
	 */
	protected void buildInputLayer() {
		
		//we take layer at 1st position of the array as input layer
		this.layers[0] = new Layer(0, inputTerminalCount);
		
		//we can think of each input terminal as a perceptron with a single
		//input and a single output with no non-linear function
		for(int i = 0; i < inputTerminalCount; i++) {
			this.layers[0].addPerceptron(i, new InputTerminal());			
		}
	}
	
	/**
	 * This is how we build hidden layers
	 */
	protected void buildHiddenLayers() {
		
		//build each hidden layer and output layer
		for(int i = 0; i < hiddenLayerCount; i++) {
			
			// the number of the perceptrons in the current hidden layer
			int currLayerSize = hiddenLayerSizes[i];
			
			//we have the input layer in the 0th position in layers array
			this.layers[i+1] = new Layer(i+1, currLayerSize);
			
			//size of previous layer
			int prevLayerSize = this.layers[i].getPerceptronCount();
			
			//set up perceptrons for this hidden layer
			for(int j = 0; j < currLayerSize; j++) {
				
				// generate a perceptron with 0.2f threshold and sigmoid
				// activation function
				Perceptron currentPerceptron = new Perceptron(prevLayerSize, 
						ActivationFunction.SIGMOID, this.commonThreshold);
				// assign a random weight for each of the inputs
				for(int k = 0; k < prevLayerSize; k++) {
					float weight = MathUtils.getBoundedRandom(-1.0f, 1.0f);
					currentPerceptron.setWeight(k, weight);
				}
				this.layers[i+1].addPerceptron(j, currentPerceptron);				 
			}
		}
	}
	
	/**
	 * This is how we build the output layer
	 */
	protected void buildOutputLayer() {
		
		//obtain the size of the hidden layer just before the output layer
		int lastHiddenLayerSize = this.layers[this.hiddenLayerCount].getPerceptronCount();
		
		//we take layer at last position of  the array as output layer
		int outputLayerIndex = (this.hiddenLayerCount + 2) - 1; 
		this.layers[outputLayerIndex] = new Layer(outputLayerIndex, outputTerminalCount);
		
		//an output perceptron is same as a perceptron except that it has a 
		//desired output value
		for(int i = 0; i < outputTerminalCount; i++) {
			
			// create perceptrons for output layer 
			OutputPerceptron currentPerceptron = new OutputPerceptron(lastHiddenLayerSize,
					ActivationFunction.SIGMOID, this.commonThreshold);
			for(int j = 0; j < lastHiddenLayerSize; j++) {
				// assign random weights for each perceptron
				float weight = MathUtils.getBoundedRandom(-1.0f, 1.0f);
				currentPerceptron.setWeight(j, weight);
			}
			this.layers[outputLayerIndex].addPerceptron(i, currentPerceptron);			
		}
	}
	
	/**
	 * This is the core backpropagation algorithm.
	 * How we train our neural network with respect to a single
	 * input versus its desired output.
	 * @param inputs
	 * @param desiredOutputs
	 */
	protected void trainForSingleInput(boolean[] inputs, boolean[] desiredOutputs) {
		
		if (inputs.length == inputTerminalCount
				&& desiredOutputs.length == outputTerminalCount) {
			
			Layer inputLayer = this.layers[0];
			Layer outputLayer = this.layers[(this.hiddenLayerCount + 2) - 1];

			// set up input for the current training datum
			for (int i = 0; i < inputs.length; i++) {
				InputTerminal inputTerminal = 
					(InputTerminal) inputLayer.getPerceptron(i);
				inputTerminal.setInput(inputs[i]);
			}

			// set up desired output for current training datum
			for (int j = 0; j < desiredOutputs.length; j++) {
				// output layer is found at last in the layers array
				OutputPerceptron outputPerceptron = 
					(OutputPerceptron) outputLayer.getPerceptron(j);
				outputPerceptron.setDesiredOutput(desiredOutputs[j]);
			}
			
			// calculate outputs for each perceptron from the 2nd layer onwards
			// this is the forward pass calculating all the outputs and
			// storing them in each perceptron 
			// based on weighted sum and activation function and threshold
			// for each perceptron.
			for(int k = 1; k < this.layers.length; k++) {
				Layer prevLayer = this.layers[k-1];
				Layer currLayer = this.layers[k];
				int prevLayerSize = prevLayer.getPerceptronCount();
				int currLayerSize = currLayer.getPerceptronCount();
				
				// obtain each perceptron of the current layer
				for(int l = 0; l < currLayerSize; l++) {
					Perceptron currLayerPerceptron = currLayer.getPerceptron(l);
					
					//set up values for inputs of the current perceptron 
					//by obtaining stored outputs in the previous layer
					for(int m = 0; m < prevLayerSize; m++) {
						Perceptron prevLayerPerceptron = prevLayer.getPerceptron(m);
						currLayerPerceptron.setInput(m, prevLayerPerceptron.getStoredOutput());
					}
					
					//calculate output for the current perceptron
					//in the current layer and store the output
					//within the perceptron for latter use
					currLayerPerceptron.calculateOutput();
				}
			} //we have finished calculating outputs for each perceptrons
			//in each layer at this point
			
			//calculate the delta values for each perceptron of the output layer 
			//based on their (desired output - actual output)
			for(int i = 0; i < this.outputTerminalCount; i++) {
				OutputPerceptron outputPerceptron = 
					(OutputPerceptron)outputLayer.getPerceptron(i);
				boolean desiredOutput = outputPerceptron.getDesiredOutput();
				boolean calculatedOutput = outputPerceptron.getStoredOutput(); 
				float delta = 
					MathUtils.booleanToFloat(desiredOutput) - MathUtils.booleanToFloat(calculatedOutput);
				outputPerceptron.setDelta(delta);
			}
			
			// now is the backward pass to calculate all delta values
			// for perceptrons other than in the output layer
			for(int j = this.layers.length - 2; j > 0; j--) {
				
				Layer currentLayer = this.layers[j];
				Layer nextLayer = this.layers[j+1];
				int currLayerSize = currentLayer.getPerceptronCount();
				int nextLayerSize = nextLayer.getPerceptronCount();
				
				//so we need to find deltas for each perceptron
				//in the current layer 
				//based on deltas of perceptrons in next layer
				for(int k = 0; k < currLayerSize; k++) {
					
					// Obtain a perceptron for the current layer
					Perceptron currLayerPerceptron = currentLayer.getPerceptron(k);
					// Calculate weighted delta value based on perceptrons in next layer
					float delta = 0.0f;
					for(int l = 0; l < nextLayerSize; l++) {
						Perceptron nextLayerPerceptron = nextLayer.getPerceptron(l);
						float kthweightOflthPerceptron = nextLayerPerceptron.getWeight(k);
						float deltaOflthPerceptron = nextLayerPerceptron.getDelta();
						delta += kthweightOflthPerceptron * deltaOflthPerceptron;						
					}
					
					// Assign the calculated delta value
					currLayerPerceptron.setDelta(delta);
				}
				
			} //so we have calculated delta values for all perceptrons
			// expect input terminals
			// (we don't need a delta value for input terminals)
			
			//the last step is to update the weights in the network based on the
			//delta values
			for(int i = 1; i < this.layers.length; i++) {
				
				// obtain the current layer
				Layer currLayer = this.layers[i];
				int currLayerSize = currLayer.getPerceptronCount();
				
				// for each perceptron update the input weights
				for(int j = 0; j < currLayerSize; j++) {
					
					Perceptron currLayerPerceptron = currLayer.getPerceptron(j);
					float currDelta = currLayerPerceptron.getDelta();
					float diffActivationFunctionValue = currLayerPerceptron.getDifferentialOutput();
					
					for(int k = 0; k < currLayerPerceptron.getInputSize(); k++) {
						float currWeight = currLayerPerceptron.getWeight(k);
						float currInput = 
							MathUtils.booleanToFloat(currLayerPerceptron.getInput(k));
						
						// calculate the new weight and update the 
						// new weight in the perceptron
						float newWeight = currWeight + 
							this.learningRate * currDelta * diffActivationFunctionValue * currInput;
						currLayerPerceptron.setWeight(k, newWeight);
					}
				}
			}
			// whooo !!!
			// finally we have finished updating the weights of the neural network 
			// with respect to this specific input
			
		}
		
	}
	
	/**
	 * This is a convenient method that can be used to dump the output of
	 * our neural network to an output file after the completion of
	 * each iteration.
	 * @param outputFileName
	 * @param iteration
	 * @throws Exception
	 */
	public void dumpNeuralNetToFile(String outputFileName, int iteration) throws Exception {
		
		File outputFile = new File(outputFileName); 
		BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile, true));
		
		writer.write("\r\n\r\n============================\r\n");
		writer.write(Integer.toString(this.inputTerminalCount) + " inputs, ");
		writer.write(Integer.toString(this.outputTerminalCount) + " outputs, ");
		writer.write(Integer.toString(this.hiddenLayerCount) + " hidden layers \r\n");
		writer.write("Iteration " + Integer.toString(iteration) + "\r\n\r\n");
				
		for(int i = 1; i < this.layers.length; i++) {
			
			Layer currLayer = this.layers[i];
			int currLayerSize = currLayer.getPerceptronCount();
			
			writer.write("Layer " + Integer.toString(i) + " with " + currLayerSize + " perceptrons\r\n\r\n");
			
			for(int j = 0; j < currLayerSize; j++) {
				Perceptron currPerceptron = currLayer.getPerceptron(j);
				writer.write("Perceptron " + Integer.toString(j) + " with " + currPerceptron.getInputSize() + " inputs. \r\n");
				writer.write("Current input weights are : ");
				for(int k = 0; k < currPerceptron.getInputSize(); k++) {
					writer.write(Float.toString(currPerceptron.getWeight(k)) + " ");
				}
				writer.write("\r\n");
			}
			
			writer.write("\r\n\r\n");
		}
		
		writer.close();
	}
	
	/**
	 * This is the method we should execute to trigger the training process.
	 * @param outputFile
	 * @param iterations
	 * @param inputVectors
	 * @param desiredOutputVectors
	 */
	public void trainNeuralNet(String outputFile, 
			int iterations, 
			boolean[][] inputVectors, 
			boolean[][] desiredOutputVectors) {
		
		try {
			for(int i = 0; i < iterations; i++) {
				this.trainForSingleInput(
						inputVectors[i % inputVectors.length], 
						desiredOutputVectors[i % desiredOutputVectors.length]);
				
				this.dumpNeuralNetToFile(outputFile, i);
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
		}
	}
}
