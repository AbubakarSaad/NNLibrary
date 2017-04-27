package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;

import Library.functions.Functions;

/**
 * This class has the main feed forward logic and implementation. This class
 * is called and used in the training techniques class for the various training
 * techniques.
 * @author Sulman and Abubakar
 */
public class FeedForward {

	private INDArray hiddenLayerWeights;
	private INDArray outputLayerWeights;
	private INDArray biasArrayH;
	private INDArray biasArrayO;
	private Functions sig = new Functions();
	private INDArray dotproductatHidden;
	private INDArray hiddenLayerSValues;
	private INDArray dotproductatOutput;
	private INDArray outputLayerSValues;
	
	/**
	 * This is the constructor for feed forward which uses the val
	 * @param hiddenLayerWeights - holds the hiddenLayerWeights values 
	 * @param outputLayerWeights - holds the outputLayerWeights values
	 */
	public FeedForward(INDArray hiddenLayerWeights, INDArray outputLayerWeights)
	{
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
	}
	
	/**
	 * This is the constructor for feed forward including the bias
	 * @param hiddenLayerWeights - holds the hiddenLayerWeights weights
	 * @param outputLayerWeights - holds the outputLayerWeights weights
	 * @param biasArrayH - holds the bias values for hidden layer.
	 * @param biasArrayO - holds the bias values for output layer.
	 */
	public FeedForward(INDArray hiddenLayerWeights, INDArray outputLayerWeights, INDArray biasArrayH, INDArray biasArrayO)
	{
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		this.biasArrayH = biasArrayH;
		this.biasArrayO = biasArrayO;
	}
	
	/**
	 * This method does the calculations for feed forward on the network
	 * @param row - holds a row of the training data.
	 */
	public void forwardPass(INDArray row)
	{

		dotproductatHidden = row.mmul(hiddenLayerWeights);
		// bias require here for hidden layer
		
		if(biasArrayH != null)
		{
			if(biasArrayH.size(0) > 0) dotproductatHidden = dotproductatHidden.add(biasArrayH);
			
		}
		//if(biasArray.length() > 0) 
		
		// Sigmoided Values
		hiddenLayerSValues = sig.sigmoid(dotproductatHidden, false);
		
		// dotproduct for summation of the hiddenLayerWeights and weights (hiddenLayer to outputLayerWeights)
		dotproductatOutput = hiddenLayerSValues.mmul(outputLayerWeights);
		
		// bias for output layer 
		if(biasArrayO != null)
		{
			if(biasArrayO.size(0) > 0) dotproductatHidden = dotproductatHidden.add(biasArrayO);
		}
		// output layer sigmoided values
		outputLayerSValues = sig.sigmoid(dotproductatOutput, false);
	}
	
	/**
	 * This method returns output at the Hidden Layer
	 * @return hiddenLayerSValues - holds the hidden layer output
	 */
	public INDArray getOutputofHiddenLayer()
	{
		return hiddenLayerSValues;
	}
	
	/**
	 * This method returns the output at the output layer
	 * @return outputLayerSValues - holds the output layer output
	 */
	public INDArray getOutputofOutputLayer()
	{
		return outputLayerSValues;
	}
}
