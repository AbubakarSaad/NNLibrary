package Library.test;
import Library.neuralnetwork.NeuralNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;


public class Main {
	
	private INDArray hiddenLayerOutput;
	private INDArray outputLayerOutput;
    
	public Main()
    {
		// parameters order: hiddenLayerSize, outputLayerSize, epochs. 
		NeuralNetwork NN = new NeuralNetwork(10, 10, 100);
		NN.holdoutTraining();

    }
    
    public static void main( String[] args )
    {
        new Main();
    }
}
