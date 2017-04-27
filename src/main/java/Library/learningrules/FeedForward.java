package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Dot;

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
	private INDArray hiddenLayerSValues;
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

		INDArray inputLayer = row;
		
		INDArray dotproductH = inputLayer.mmul(hiddenLayerWeights);
		INDArray sumh = dotproductH.add(biasArrayH);

		hiddenLayerSValues = sig.sigmoid(sumh, false);
		
		INDArray dotproductO = hiddenLayerSValues.mmul(outputLayerWeights);
		INDArray sumo = dotproductO.add(biasArrayO);

		
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
