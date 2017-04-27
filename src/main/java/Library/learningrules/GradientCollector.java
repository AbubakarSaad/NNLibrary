package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
	
	public GradientCollector(INDArray hiddenLayerWeights, INDArray outputLayerWeights)
	{
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		gradientForHidden =  Nd4j.zeros(hiddenLayerWeights.size(0), hiddenLayerWeights.size(1));
		gradientForOutput = Nd4j.zeros(outputLayerWeights.size(0), outputLayerWeights.size(1));
	}
	
	public void gradients(INDArray outputLayerOutput, INDArray hiddenLayerOutput, INDArray errorAtOutput, INDArray sample)
	{
		
		 outputSigmoidedValues = func.sigmoid(outputLayerOutput, true);
		 errorContrAtOutput = outputSigmoidedValues.mul(errorAtOutput);
		 
		 
		 INDArray errorContrAtOutputT = errorContrAtOutput.transpose();
		 
		 gradientForOutput = errorContrAtOutputT.mmul(hiddenLayerOutput);
		 gradientForOutput = gradientForOutput.transpose();
		 
		 
		 
		 
		 // The hidden layer error contributions
		 INDArray errorContrAtHidden = errorContrAtOutput.mmul(outputLayerWeights.transpose());
		 
		 INDArray bi = func.sigmoid(hiddenLayerOutput, true);
		 INDArray errorAtHidden = errorContrAtHidden.mul(bi);
		 
		 
		 INDArray errAtHidden = errorAtHidden.transpose();
		 gradientForHidden = errAtHidden.mmul(sample);
		 gradientForHidden = gradientForHidden.transpose();
	}
	
	
	public INDArray getGradientForHidden()
	{
		return gradientForHidden;
	}
	
	public INDArray getGradientForOutput()
	{
		return gradientForOutput;
	}
}
