package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;

import Library.functions.Functions;

public class FeedForward {

	private INDArray hiddenlayer;
	private INDArray outputlayer;
	private INDArray biasArrayH;
	private INDArray biasArrayO;
	private Functions sig = new Functions();
	private INDArray dotproductatHidden;
	private INDArray hiddenLayerSValues;
	private INDArray dotproductatOutput;
	private INDArray outputLayerSValues;
	
	/**
	 * This is the constructer for feedforward
	 * @param hiddenLayer - holds the hiddenlayer 
	 * @param outputLayer - holds the outputlayer
	 */
	public FeedForward(INDArray hiddenLayer, INDArray outputLayer)
	{
		this.hiddenlayer = hiddenLayer;
		this.outputlayer = outputLayer;
	}
	
	/**
	 * This is the contructor for feedforward including the bias
	 * @param hiddenLayer - holds the hiddenLayer weights
	 * @param outputLayer - holds the outputLayer weights
	 * @param biasArrayH - holds the bias values for hidden layer
	 * @param biasArrayO - holds the bias values for output layer
	 */
	public FeedForward(INDArray hiddenLayer, INDArray outputLayer, INDArray biasArrayH, INDArray biasArrayO)
	{
		this.hiddenlayer = hiddenLayer;
		this.outputlayer = outputLayer;
		this.biasArrayH = biasArrayH;
		this.biasArrayO = biasArrayO;
	}
	
	/**
	 * This method does the calculations for feed forward on the network
	 * @param row - holds an input of the data 
	 */
	public void forwardPass(INDArray row)
	{

		dotproductatHidden = row.mmul(hiddenlayer);
		// bias require here for hidden layer
		
		if(biasArrayH != null)
		{
			if(biasArrayH.size(0) > 0) dotproductatHidden = dotproductatHidden.add(biasArrayH);
			
		}
		
		
		//if(biasArray.length() > 0) 
		
		// Sigmoided Values
		hiddenLayerSValues = sig.sigmoid(dotproductatHidden, false);
		
		// dotproduct for summation of the hiddenlayer and weights (hiddenLayer to outputlayer)
		dotproductatOutput = hiddenLayerSValues.mmul(outputlayer);
		
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
