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
	private INDArray deltaArrayH;
	private INDArray deltaArrayO;
	private boolean momentum;
	private double momentumV;
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
	 * The third constructor used when momentum and bias is implement for each layer
	 * @param hiddenLayerWeights - holds hidden Layer weights
	 * @param outputLayerWeights - holds output layer weights
	 * @param learningRate - learning rate
	 * @param biasArrayH - bias for hidden layer
	 * @param biasArrayO - bias for output layer 
	 * @param momentum - momentum, if used by user 
	 * @param momentumV - momentum value
	 */
	public Backprop(INDArray hiddenLayerWeights, INDArray outputLayerWeights, double learningRate, INDArray biasArrayH, INDArray biasArrayO, Boolean momentum, double momentumV)
	 {
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		this.biasArrayH = biasArrayH;
		this.biasArrayO = biasArrayO;
		this.learningRate = learningRate;
		deltabiasArrayH = Nd4j.zeros(1, this.biasArrayH.size(1));
		deltabiasArrayO = Nd4j.zeros(1, this.biasArrayO.size(1));
		deltaArrayH = Nd4j.zeros(hiddenLayerWeights.size(0), hiddenLayerWeights.size(1));
		deltaArrayO = Nd4j.zeros(outputLayerWeights.size(0), outputLayerWeights.size(1));
		this.momentum = momentum;
		this.momentumV = momentumV;
		
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
		 INDArray deltao2;
		 if (momentum == true)
		 {
			 INDArray deltaom = deltaArrayO.mul(momentumV);
			 deltao2 = outputLayerWeights.sub(deltao1.add(deltaom));
		 }else {
			 deltao2 = outputLayerWeights.sub(deltao1);
		 }
		
	
		 if(biasArrayO != null){
			 INDArray deltaob = erroratoutputLayer.mul(learningRate);
			 INDArray deltaob2 = biasArrayO.sub(deltaob);
			 deltabiasArrayO.assign(deltaob2);
		 }
		 
		 
		 INDArray errorContr = erroratoutputLayer.mmul(outputLayerWeightsCopy.transpose());
		 INDArray bi = func.sigmoid(hiddenLayerOutput, true);
		 INDArray errorHiddenLayer = errorContr.mul(bi);
		 
		 INDArray errHiddenLayer = errorHiddenLayer.transpose();
		 INDArray deltah = (errHiddenLayer.mmul(data)).transpose();
		 INDArray deltah1 = deltah.mul(learningRate);
		 INDArray deltah2;
		 if(momentum == true)
		 {
			 INDArray deltahm = deltaArrayH.mul(momentumV);
			 deltah2 = hiddenLayerWeights.sub(deltah1.add(deltahm));
		 }else{
			 deltah2 = hiddenLayerWeights.sub(deltah1);
		 }
		 hiddenLayerWeights.assign(deltah2);
		 
		 if(biasArrayH != null){
			 INDArray deltaoh = errorHiddenLayer.mul(learningRate);
			 INDArray deltaoh2 = biasArrayH.sub(deltaoh);
			 deltabiasArrayH.assign(deltaoh2);
		 }
		 
		 outputLayerWeights.assign(deltao2);

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
