package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;

import Library.functions.Functions;

public class GraidentCollector {

	private INDArray outputSigmoidedValues;
	private INDArray errorContrAtOutput;
	Functions func = new Functions();
	private INDArray hiddenLayerWeights;
	private INDArray outputLayerWeights;
	private INDArray gradientForOutput;
	private INDArray gradientForHidden;
	
	public GraidentCollector()
	{
		
	}
	
	public void graidents(INDArray outputLayerOutput, INDArray hiddenLayerOutput, INDArray errorAtOutput, INDArray sample)
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
