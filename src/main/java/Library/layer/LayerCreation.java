package Library.layer;

import java.util.Random;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Abu on 4/23/2017.
 */
public class LayerCreation {

    private INDArray weights;
    
    /**
     * This method initializes the neuron class
     * @param numberofInputs - number of inputs 
     * @param numberofHidden - number of neurons in the layer
     */
    public LayerCreation(int numberofInputs, int numberofHidden)
    {
    	Nd4j.setDataType(DataBuffer.Type.FLOAT);
        // Nd4j.rand(rows, columns, min, max, random)
    	long[][] matrix = new long[numberofInputs][numberofHidden];
    	for(int i=0; i<matrix.length; i++)
    	{
    		for(int j=0; j<matrix[i].length; j++)
    		{
    			Random rand = new Random();
    			matrix[i][j] = (long) (rand.nextLong() * (0.5 - (-0.5)) + (-0.5));
    		}
    	}
    	
        weights = Nd4j.rand(numberofInputs, numberofHidden, (-0.5), 0.5, Nd4j.getRandom());
        
    }
    
    /**
     * Getter method to return weights 
     * @return - returns the layer and connections (weights)
     */
    public INDArray getWeights()
    {
        return weights;
    }
}
