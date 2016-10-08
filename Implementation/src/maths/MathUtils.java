package maths;

import java.util.Random;

/**
 * This class contains the utility functions
 * @author wimal perera (09/10008)
 *
 */
public class MathUtils {
	
	/**
     * Our random number generator.
     */
    private final static Random _random = new Random();
	
	/**
     * Return a random number within the specified range
     * 
     * @param lower range
     * @param upper range
     * @return a random number within the specified range
     */
    public static synchronized float getBoundedRandom(float lower, float upper) {
    	float range = upper - lower;
    	float result = _random.nextFloat() * range + lower;
    	return(result);
    }
    
    /**
     * This is the sigmoid function.
     * @param x
     * @return
     */
    public static float sigmoid(float x) {
    	return (float) (1.0f / (1.0f + Math.exp(-1.0f * x)));
    }
    
    /**
     * This is the differentiated sigmoid function.
     * @param x
     * @return
     */
    public static float diffSigmoid(float x) {
    	return (float) (Math.exp(-1.0f * x) / Math.pow(1 + Math.exp(-1.0 * x), 2));
    }
    
    /**
     * A convenient method between switching from booleans to floats
     * @param value
     * @return
     */
    public static float booleanToFloat(boolean value) {
    	if(value)
    		return 1.0f;
    	else
    		return 0.0f;
    }
    
    /**
     * A convenient method which can be used to write a boolean array
     * to the output
     * @param array
     * @return
     */
    public static String booleanArrayToString(boolean[] array) {
    	
    	String outputString = "";
    	for(int i = 0; i < array.length; i++) {
    		outputString += array[i] ? "1" : "0";
    	}
    	return outputString;
    }
}
