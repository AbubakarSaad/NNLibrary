package Library;
import org.nd4j.linalg.api.ndarray.INDArray;


public class Main 
{
	public int epoch = 1;
	private INDArray hiddenLayerOutput;
	private INDArray outputLayerOutput;
    
	public Main()
    {
		
		
		
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
