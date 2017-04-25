package Library;

import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import Library.neuron.Neuron;
import learningrul.FeedForward;


public class App 
{
	public int epoch = 1;
	private INDArray hiddenLayerOutput;
	private INDArray outputLayerOutput;
    
	public App()
    {
		final String dir = System.getProperty("user.dir");
		    	
		// input layer 
		INDArray inputLayer = getInputLayer(dir + "//a1digits//digit_test_0.txt", ",");
		//System.out.print(inputLayer);
		
		// neuron(sizeofInput, numberofHidden)
		INDArray hiddenLayer = getHiddenLayer(inputLayer.size(1), 2);
		System.out.println(hiddenLayer);
		
		INDArray outputLayer = getOutputLayer(hiddenLayer.size(1), 10);
		
		FeedForward ff = new FeedForward(hiddenLayer, outputLayer);
		
		
		
		// send one sample at a time
		for(int i=0; i<epoch; i++)
		{
			System.out.println("----------------------Epoch: " + i + "--------------------------");
			for(int j=0; j<inputLayer.size(0) - 399; j++)
			{
				ff.caluclations(inputLayer.getRow(j));
				
				hiddenLayerOutput = ff.getOutputofHiddenLayer();
				outputLayerOutput = ff.getOutputofOutputLayer();
			
			}
		}
    }
    
	/**
	 * 
	 * @param fileName
	 * @param delimiter
	 * @return
	 */
    public INDArray getInputLayer(String fileName, String delimiter)
	{
		try {
			return Nd4j.readNumpy(fileName, delimiter);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
    /**
     * 
     * @param inputlayerSize
     * @param numOfHiddenNeurons
     * @return
     */
	public INDArray getHiddenLayer(int inputlayerSize, int numOfHiddenNeurons)
	{
		
		return new Neuron(inputlayerSize, numOfHiddenNeurons).getWeights();
	}
	
	/**
	 * 
	 * @param hiddenLayerSize
	 * @param numOfOutputNeurons
	 * @return
	 */
	public INDArray getOutputLayer(int hiddenLayerSize, int numOfOutputNeurons)
	{
		return new Neuron(hiddenLayerSize, numOfOutputNeurons).getWeights();
	}


    public static void main( String[] args )
    {
        new App();
        System.out.println( "Hello World!" );
    }
}
