package Library.test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

import Library.neuralnetwork.NeuralNetwork;

/**
 * This class is the test class that calls on the library tools and creates
 * the neural network.
 * @authors Sulman and Abubakar
 */
public class Main {
	public Main()
    {

	// parameters order: hiddenLayerSize, outputLayerSize, epochs, learning rate, bias.	
		NeuralNetwork NNB = new NeuralNetwork(25, 10, 30, 0.05, "bias", true, 0.2);
		NNB.holdoutTraining();
		
		//NNB.deltabardelta();
		//NNB.rprop(



		
    }
    
    public static void main( String[] args )
    {
    	Nd4j.setDataType(DataBuffer.Type.FLOAT);
        new Main();
    }
}
