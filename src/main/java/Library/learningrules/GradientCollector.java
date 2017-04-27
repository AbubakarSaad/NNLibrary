package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;

import Library.functions.Functions;

/**
 * This class is a general class that is based of the backprop class but 
 * this only calculates the gradients. This method is used in conjunction with
 * rProp and delta-bar-delta classes.
 * @author Sulman and Abubakar
 */
public class GradientCollector {

	private INDArray outputSigmoidedValues;
	private INDArray errorContrAtOutput;
	Functions func = new Functions();
	private INDArray hiddenLayerWeights;
	private INDArray outputLayerWeights;
	private INDArray gradientForOutput;
	private INDArray gradientForHidden;
	
	public GradientCollector()
	{
		
	}
	
	public void gradients(INDArray outputLayerOutput, INDArray hiddenLayerOutput, INDArray errorAtOutput, INDArray sample)
	{
		 outputSigmoidedValues = func.sigmoid(outputLayerOutput, true);
		 errorContrAtOutput = outputSigmoidedValues.muli(errorAtOutput);
		 
		 
		 INDArray errorContrAtOutputT = errorContrAtOutput.transpose();
		 
		 gradientForOutput = errorContrAtOutputT.mmul(hiddenLayerOutput);
		 gradientForOutput = gradientForOutput.transpose();
		 
		 
		 
		 
		 // The hidden layer error contributions
		 INDArray errorContrAtHidden = errorContrAtOutput.mmul(outputLayerWeights.transpose());
		 
		 INDArray bi = func.sigmoid(hiddenLayerOutput, true);
		 INDArray errorAtHidden = errorContrAtHidden.muli(bi);
		 
		 
		 errorAtHidden = errorAtHidden.transpose();
		 
		 gradientForHidden = errorAtHidden.mmul(sample);
		 gradientForHidden = gradientForHidden.transpose();
		 
		 
	}
}
