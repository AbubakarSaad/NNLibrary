package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Dot;

import Library.functions.Functions;

public class FeedForward {

	private INDArray hiddenlayerWeights;
	private INDArray outputlayerWeights;
	private INDArray biasArrayH;
	private INDArray biasArrayO;
	private Functions sig = new Functions();
	private INDArray hiddenLayerSValues;
	private INDArray outputLayerSValues;
	
	/**
	 * This is the constructer for feedforward
	 * @param hiddenLayer - holds the hiddenlayer 
	 * @param outputLayer - holds the outputlayer
	 */
	public FeedForward(INDArray hiddenLayer, INDArray outputLayer)
	{
		this.hiddenlayerWeights = hiddenLayer;
		this.outputlayerWeights = outputLayer;
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
		this.hiddenlayerWeights = hiddenLayer;
		this.outputlayerWeights = outputLayer;
		this.biasArrayH = biasArrayH;
		this.biasArrayO = biasArrayO;
	}
	
	/**
	 * This method does the calculations for feed forward on the network
	 * @param row - holds an input of the data 
	 */
	public void forwardPass(INDArray row)
	{
		INDArray inputLayer = row;
		
		INDArray dotproductH = inputLayer.mmul(this.hiddenlayerWeights);
		INDArray sumh = dotproductH.add(this.biasArrayH);
		
		hiddenLayerSValues = sig.sigmoid(sumh, false);
		
		INDArray dotproductO = hiddenLayerSValues.mmul(this.outputlayerWeights);
		INDArray sumo = dotproductO.add(this.biasArrayO);
		
		outputLayerSValues = sig.sigmoid(sumo, false);
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
