package Library.learningrules;

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
	
	public Backprop(INDArray hiddenLayerWeights, INDArray outputLayerWeights, double learningRate, INDArray biasArrayH, INDArray biasArrayO)
	 {
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		this.biasArrayH = biasArrayH;
		this.biasArrayO = biasArrayO;
		this.learningRate = learningRate;
	 }
	 
	 /**
	  * This method calculatues the gradients, updates bias and the weights for layers  
	  * @param outputLayerOutput
	  * @param hiddenLayerOutput
	  * @param errorAtOutput
	  * @param sample
	  */
	 public void calculations(INDArray outputLayerOutput, INDArray hiddenLayerOutput, INDArray errorAtOutput, INDArray sample)
	 {
		 // The output layer error contributions
		 outputSigmoidedValues = func.sigmoid(outputLayerOutput, true);
		 
		 errorContrAtOutput = outputSigmoidedValues.muli(errorAtOutput);
		 
		 
		 INDArray errorContrAtOutputT = errorContrAtOutput.transpose();
		 
		 gradientForOutput = errorContrAtOutputT.mmul(hiddenLayerOutput);
		 gradientForOutput = (gradientForOutput.transpose()).muli(learningRate);
		 
		 // update the weights
		 outputLayerWeights = outputLayerWeights.sub(gradientForOutput);
		 
		 // update the bias of output layer
		 if(biasArrayH != null){
			 INDArray deltaBiasH = errorContrAtOutput.muli(learningRate);
			 biasArrayH = biasArrayH.sub(deltaBiasH);
		 }
		 
		 // The hidden layer error contributions
		 INDArray errorContrAtHidden = errorContrAtOutput.mmul(outputLayerWeights.transpose());
		 
		 INDArray bi = func.sigmoid(hiddenLayerOutput, true);
		 INDArray errorAtHidden = errorContrAtHidden.muli(bi);
		 
		 
		 INDArray errorAtHiddenLayer = errorAtHidden.transpose();
		 
		 gradientForHidden = errorAtHiddenLayer.mmul(sample);
		 gradientForHidden = (gradientForHidden.transpose()).muli(learningRate);
		 
		 hiddenLayerWeights = hiddenLayerWeights.sub(gradientForHidden);
		 
		 // updating the bias for the hidden layer
		 if(biasArrayO != null){
			 INDArray deltaBiasO = errorAtHidden.muli(learningRate);
			 biasArrayO = biasArrayO.sub(deltaBiasO);
		 }
		 
		 
		 
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
	 
}
