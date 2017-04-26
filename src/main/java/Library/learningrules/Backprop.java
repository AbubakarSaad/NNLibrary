package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;

import Library.functions.Functions;

public class Backprop {

	private INDArray outputSigmoidedValues;
	private INDArray errorContrAtOutput;
	Functions func = new Functions();
	private INDArray hiddenLayerWeights;
	private INDArray outputLayerWeights;
	
	public Backprop(INDArray hiddenLayerWeights, INDArray outputLayerWeights)
	 {
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
	 }
	 
	 
	 public void calculations(INDArray outputLayerOutput, INDArray hiddenLayerOutput, INDArray errorAtOutput, INDArray sample)
	 {
	  
		 // The output layer error contributions
		 outputSigmoidedValues = func.sigmoid(outputLayerOutput, true);
		 errorContrAtOutput = outputSigmoidedValues.muli(errorAtOutput);
		 
		 INDArray errorContrAtOutputT = errorContrAtOutput.transpose();
		 
		 System.out.println(errorContrAtOutputT);
		 System.out.println(hiddenLayerOutput);
		 
		 INDArray gradientForOutput = errorContrAtOutputT.mmul(hiddenLayerOutput);
		 System.out.println("gradientForOutput: \n" + gradientForOutput);
		 
		 // delta === gradient
		 gradientForOutput = gradientForOutput.transpose();
		 System.out.println("gradient Transposed back: \n" + gradientForOutput);
		 
		 // The hidden layer error contributions
		 INDArray errorContrAtHidden = errorContrAtOutput.mmul(outputLayerWeights.transpose());
		 
		 INDArray bi = func.sigmoid(hiddenLayerOutput, true);
		 INDArray errorAtHidden = errorContrAtHidden.muli(bi);
		 
		 
		 errorAtHidden = errorAtHidden.transpose();
		 
		 INDArray gradientForHidden = errorAtHidden.mmul(sample);
		 gradientForHidden = gradientForHidden.transpose();
		 
		 System.out.println("gradient for hidden: \n" + gradientForHidden);
		 
		 
	 }
}
