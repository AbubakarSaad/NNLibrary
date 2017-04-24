package Library;

import java.io.IOException;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import Library.neuron.Neuron;
import learningrul.FeedForward;

/**
 * Hello world!
 *
 */
public class App 
{
	public int epoch = 1;
    public App()
    {
        Neuron n = new Neuron(2, 3);
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
			}
		}
    }
    
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
	
	public INDArray getHiddenLayer(int inputlayerSize, int numOfHiddenNeurons)
	{
		
		return new Neuron(inputlayerSize, numOfHiddenNeurons).getWeights();
	}
	
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
