package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;

import Library.functions.Functions;

public class Backprop {

	private INDArray outputSigmoidedValues;
	private INDArray errorContrAtOutput;
	Functions func = new Functions();
	
	public Backprop(INDArray hiddenLayer, INDArray outputLayer)
	 {
	  
	 }
	 
	 
	 public void calculations(INDArray outputLayerOutput, INDArray hiddenLayerOutput, INDArray errorAtOutput, INDArray sample)
	 {
	  
	  outputSigmoidedValues = func.sigmoid(outputLayerOutput, true);
	  errorContrAtOutput = outputSigmoidedValues.muli(errorAtOutput);
	 }
}
