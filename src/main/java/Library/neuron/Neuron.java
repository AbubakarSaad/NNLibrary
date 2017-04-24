package Library.neuron;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Abu on 4/23/2017.
 */
public class Neuron {

    private INDArray weights;
    
    /**
     * This method initailizes the neuron class
     * @param numberofInputs - number of inputs 
     * @param numberofHidden - number of neurons in the layer
     */
    public Neuron(int numberofInputs, int numberofHidden)
    {

        //System.out.print(Nd4j.getRandom());
        // Nd4j.rand(rows, columns, min, max, random)
        weights = Nd4j.rand(numberofInputs, numberofHidden, -0.5, 0.5, Nd4j.getRandom());
        

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
