package Library.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import Library.functions.Functions;
import Library.layer.LayerCreation;
import Library.learningrules.Backprop;
import Library.learningrules.FeedForward;

/**
 * Created by Abu on 4/23/2017.
 */
public class Test {
	
	final String dir = System.getProperty("user.dir");
	public int epoch = 1;
	
	Functions func = new Functions();
	private INDArray hiddenLayerOutput;
	private INDArray outputLayerOutput;
    public Test()
    {
        INDArray nd = Nd4j.create(new float[]{1,2}, new int[]{2}); // row vectors
        INDArray nd2 = Nd4j.create(new float[]{3,4,5,6}, new int[]{2,2}); //column vector 
        
        List<INDArray> list = new ArrayList<INDArray>();
        list.add(new LayerCreation(2, 3).getWeights());
        list.add(new LayerCreation(2, 3).getWeights());
        INDArray moreHiddenLayers = Nd4j.create(list, new int[]{2,2});
        
        INDArray inputLayer = getInputLayer(dir + "//a1digits//digit_test_0.txt", ",");
		//System.out.print(inputLayer);
		
		// neuron(sizeofInput, numberofHidden)
		INDArray hiddenLayer = getHiddenLayer(inputLayer.size(1), 2);
		//System.out.println(hiddenLayer);
		
		INDArray outputLayer = getOutputLayer(hiddenLayer.size(1), 10);
		
		FeedForward ff = new FeedForward(hiddenLayer, outputLayer);
		
		Backprop bp = new Backprop(hiddenLayer, outputLayer);
		
		for(int i=0; i<epoch; i++)
		{
			System.out.println("----------------------Epoch: " + i + "--------------------------");
			for(int j=0; j<inputLayer.size(0) - 399; j++)
			{
				ff.feedForward(inputLayer.getRow(j));
				
				hiddenLayerOutput = ff.getOutputofHiddenLayer();
				outputLayerOutput = ff.getOutputofOutputLayer();
				
				
				INDArray errorAtOutput = outputLayerOutput.sub(bp.exceptedOutput(0));
			
				bp.calculations(outputLayerOutput, hiddenLayerOutput, errorAtOutput, inputLayer.getRow(j));
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
   	
       /**
        * 
        * @param inputlayerSize - size of the input layer 
        * @param numOfHiddenNeurons
        * @return
        */
   	public INDArray getHiddenLayer(int inputlayerSize, int numOfHiddenNeurons)
   	{
   		
   		return new LayerCreation(inputlayerSize, numOfHiddenNeurons).getWeights();
   	}
   	
   	/**
   	 * 
   	 * @param hiddenLayerSize
   	 * @param numOfOutputNeurons
   	 * @return
   	 */
   	public INDArray getOutputLayer(int hiddenLayerSize, int numOfOutputNeurons)
   	{
   		return new LayerCreation(hiddenLayerSize, numOfOutputNeurons).getWeights();
   	}
    public static void main(String[] args)
    {
        new Test();
    }
}
