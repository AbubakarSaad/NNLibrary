package Library.test;
import Library.neuralnetwork.NeuralNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;


public class Main {
	
	private INDArray hiddenLayerOutput;
	private INDArray outputLayerOutput;
    
	public Main()
    {
		// parameters order: hiddenLayerSize, outputLayerSize, epochs. 

//		NeuralNetwork NN = new NeuralNetwork(25, 10, 1, 0.1);
//		NN.holdoutTraining();
		
		NeuralNetwork NNB = new NeuralNetwork(25, 10, 2, 0.2, "bias");
		NNB.holdoutTraining();

		
    }
    
    public static void main( String[] args )
    {
        new Main();
    }
}
