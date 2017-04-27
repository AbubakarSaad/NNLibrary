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
	/**
	 * This is consturctor for backprop
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
	 * This constructor provides abilitly to include bias
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
	 }
	 
	 /**
	  * This method calculates the gradients 
	  * This method calculatues the gradients, updates bias and the weights for layers  
	  * @param outputLayerOutput
	  * @param hiddenLayerOutput
	  * @param errorAtOutput
	  * @param sample
	  */
	 public void calculations(INDArray outputLayerOutput, INDArray hiddenLayerOutput, INDArray errorAtOutput, INDArray sample)
	 {
		 
		 INDArray outo1neto1 = func.sigmoid(outputLayerOutput, true);
		 INDArray erroratoutputLayer = outo1neto1.mul(errorAtOutput);
		 
		 INDArray erroutputLayer = erroratoutputLayer.transpose();
		 
		 INDArray deltao = ((erroutputLayer.mmul(hiddenLayerOutput)).transpose()).mul(this.learningRate);
		 this.outputLayerWeights = this.outputLayerWeights.sub(deltao);
		 
		 INDArray deltaob = erroratoutputLayer.mul(this.learningRate);
		 this.biasArrayO = this.biasArrayO.sub(deltaob);
		 
		 INDArray errorContr = erroratoutputLayer.mmul(this.outputLayerWeights.transpose());
		 INDArray bi = func.sigmoid(errorContr, true);
		 INDArray errorHiddenLayer = errorContr.mul(bi);
		 
		 INDArray errHiddenLayer = errorHiddenLayer.transpose();
		 INDArray deltah = ((errHiddenLayer.mmul(sample)).transpose()).mul(this.learningRate);
		 this.hiddenLayerWeights = this.hiddenLayerWeights.sub(deltah);
		 
		 INDArray deltaoh = errorHiddenLayer.mul(this.learningRate);
		 this.biasArrayH = this.biasArrayH.sub(deltaoh);
		 
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
	 
	 public INDArray getBiasArrayForHidden()
	 {
		 return biasArrayH;
	 }
	 
	 public INDArray getBiasArrayForOutput()
	 {
		 return biasArrayO;
	 }
}
