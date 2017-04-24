package learningrul;

import org.nd4j.linalg.api.ndarray.INDArray;

import functions.Functions;

public class FeedForward {

	private INDArray hiddenlayer;
	private INDArray outputlayer;
	private Functions sig = new Functions();
	private INDArray dotproductatHidden;
	private INDArray hiddenLayerSValues;
	private INDArray dotproductatOutput;
	private INDArray outputLayerSValues;
	
	public FeedForward(INDArray hiddenLayer, INDArray outputLayer)
	{
		hiddenlayer = hiddenLayer;
		outputlayer = outputLayer;
	}
	
	/**
	 * 
	 * @param row
	 */
	public void caluclations(INDArray row)
	{
		System.out.println(row);
		System.out.println(hiddenlayer);
		dotproductatHidden = row.mmul(hiddenlayer);
		
		
		System.out.println("something works here");
		System.out.println(dotproductatHidden);
		
		// bias require here for hidden layer
		
		
		// Sigmoided Values
		hiddenLayerSValues = sig.sigmoid(dotproductatHidden, false);
		System.out.println(hiddenLayerSValues);
		
		// dotproduct for summation of the hiddenlayer and weights (hiddenLayer to outputlayer)
		dotproductatOutput = hiddenLayerSValues.mmul(outputlayer);
		System.out.println(dotproductatOutput);
		
		// bias for output layer 
		
		// output layer sigmoided values
		outputLayerSValues = sig.sigmoid(dotproductatOutput, false);
		System.out.println(outputLayerSValues);
	}
	
	public INDArray getOutputofHiddenLayer()
	{
		return hiddenLayerSValues;
	}
	
	
	public INDArray getOutputofOutputLayer()
	{
		return dotproductatOutput;
	}
}
