package Library.test;
import Library.neuralnetwork.NeuralNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;


public class Main {
	
	private INDArray hiddenLayerOutput;
	private INDArray outputLayerOutput;
    
	public Main()
    {
		// parameters order: hiddenLayerSize, outputLayerSize, epochs. 
		NeuralNetwork NN = new NeuralNetwork(10, 10, 3);
		
		// send one sample at a time
//		for(int i=0; i<epoch; i++)
//		{
//			System.out.println("----------------------Epoch: " + i + "--------------------------");
//			for(int j=0; j<inputLayer.size(0) - 399; j++)
//			{
//				ff.feedForward(inputLayer.getRow(j));
//				
//				hiddenLayerOutput = ff.getOutputofHiddenLayer();
//				outputLayerOutput = ff.getOutputofOutputLayer();
//			
//			}
//		}
    }
    
    public static void main( String[] args )
    {
        new Main();
    }
}
