package Library.test;
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
		NeuralNetwork NN = new NeuralNetwork(25, 10, 1, 0.2);
		NN.holdoutTraining();
		
		//NeuralNetwork NNB = new NeuralNetwork(10, 10, 1, 0.05, "bias");
		//NNB.holdoutTraining();
		
    }
    
    public static void main( String[] args )
    {
        new Main();
    }
}
