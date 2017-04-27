package Library.learningrules;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import Library.functions.Functions;

public class Backprop {
	
	private INDArray outputSigmoidedValues;
	private INDArray errorContrAtOutput;
	Functions func = new Functions();
	private INDArray hiddenLayerWeights;
	private INDArray outputLayerWeights;
	private INDArray gradientForOutput;
	private INDArray gradientForHidden;
	private INDArray biasArrayH;
	private INDArray biasArrayO;
	private double learningRate;
	private INDArray deltabiasArrayH;
	private INDArray deltabiasArrayO;
	/**
	 * This is the constructor for back propagation class which requires the
	 * hiddenlayer weights, outputlayer weights and learning rate.
	 * @param hiddenLayer
	 * @param outputLayer
	 */
	public Backprop(INDArray hiddenLayerWeights, INDArray outputLayerWeights, double learningRate)
	 {
		
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		this.learningRate = learningRate;
	 }

	/**
	 * The second constructor used when bias is implemented for each layer.
	 * @param hiddenLayerWeights
	 * @param outputLayerWeights
	 * @param learningRate
	 * @param biasArrayH
	 * @param biasArrayO
	 */
	public Backprop(INDArray hiddenLayerWeights, INDArray outputLayerWeights, double learningRate, INDArray biasArrayH, INDArray biasArrayO)
	 {
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		this.biasArrayH = biasArrayH;
		this.biasArrayO = biasArrayO;
		this.learningRate = learningRate;
		deltabiasArrayH = Nd4j.zeros(1, this.biasArrayH.size(1));
		deltabiasArrayO = Nd4j.zeros(1, this.biasArrayO.size(1));
	 }
	 
	 /**
	  * This method calculates the gradients, updates bias and the weights 
	  * for layers.  
	  * @param outputLayerOutput - values at the output layer.
	  * @param hiddenLayerOutput -  values at the hidden layer.
	  * @param errorAtOutput - the error that is found at the output layer and 
	  * propagated back.
	  * @param data - the one line of data being sent in from the training data.
	  */
	 public void calculations(INDArray outputLayerOutput, INDArray hiddenLayerOutput, INDArray errorAtOutput, INDArray data)
	 {
		 INDArray outputLayerWeightsCopy = outputLayerWeights;
		 INDArray outo1neto1 = func.sigmoid(outputLayerOutput, true);
		 INDArray erroratoutputLayer = outo1neto1.mul(errorAtOutput);
		 
		 INDArray erroutputLayer = erroratoutputLayer.transpose();
		 
		 INDArray deltao = (erroutputLayer.mmul(hiddenLayerOutput)).transpose();
		 INDArray deltao1 = deltao.mul(learningRate);
		 INDArray deltao2 = outputLayerWeights.sub(deltao1);
		 outputLayerWeights.assign(deltao2);
		 
		 
		 INDArray deltaob = erroratoutputLayer.mul(learningRate);
		 INDArray deltaob2 = biasArrayO.sub(deltaob);
		 deltabiasArrayO.assign(deltaob2);
		 
		 INDArray errorContr = erroratoutputLayer.mmul(outputLayerWeightsCopy.transpose());
		 INDArray bi = func.sigmoid(errorContr, true);
		 INDArray errorHiddenLayer = errorContr.mul(bi);
		 
		 
		 INDArray errHiddenLayer = errorHiddenLayer.transpose();
		 INDArray deltah = (errHiddenLayer.mmul(data)).transpose();
		 INDArray deltah1 = deltah.mul(learningRate);
		 INDArray deltah2 = hiddenLayerWeights.sub(deltah1);
		 hiddenLayerWeights.assign(deltah2);
		 
		 INDArray deltaoh = errorHiddenLayer.mul(learningRate);
		 INDArray deltaoh2 = biasArrayH.sub(deltaoh);
		 deltabiasArrayH.assign(deltaoh2);

	 }
	 
	 /**
	  * This method returns the updated weights for hidden layer
	  * @return - hiddenLayerWeights: holds input to hidden layer weights
	  */
	 public INDArray getUpdatedHiddenLayerWeights()
	 {
		 return hiddenLayerWeights;
	 }
	 
	 /**
	  * This method returns the updated weights for output layer
	  * @return - outputLayerWeights: holds hidden to output layer weights
	  */
	 public INDArray getUpdatedOutputLayerWeights()
	 {
		 return outputLayerWeights;
	 }
	 
	 /**
	  * This method returns the updated bias for hidden layer
	  * @return - deltabiasArrayH: holds hidden layer values
	  */
	 public INDArray getBiasArrayForHidden()
	 {
		 return deltabiasArrayH;
	 }
	 
	 /**
	  * This method returns the updated bias for output layer
	  * @return - deltabiasArrayO: holds output layer values
	  */
	 public INDArray getBiasArrayForOutput()
	 {
		 return deltabiasArrayO;
	 }
}
