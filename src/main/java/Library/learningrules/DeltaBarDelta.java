package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *This class is the main implementation of delta-bar-delta, it works in 
 * conjunction with the gradient collector class.
 * @author Sulman and Abubakar
 */
public class DeltaBarDelta {
	
	private double kGrowth;
	private double dDecay;
	private INDArray previousGradientH;
	private INDArray previousGradientO;
	private INDArray learningRateH;
	private INDArray learningRateO;
	private INDArray hiddenLayerWeights;
	private INDArray outputLayerWeights;
	
	public DeltaBarDelta(double kGrowth, double dDecay, INDArray hiddenLayerWeights, INDArray outputLayerWeights)
	{
		this.kGrowth = kGrowth;
		this.dDecay = dDecay;
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		previousGradientH = Nd4j.zeros(hiddenLayerWeights.size(0), hiddenLayerWeights.size(1));
		previousGradientO = Nd4j.zeros(outputLayerWeights.size(0), outputLayerWeights.size(1));
		learningRateH = Nd4j.ones(hiddenLayerWeights.size(0), hiddenLayerWeights.size(1));
		learningRateO = Nd4j.ones(outputLayerWeights.size(0), outputLayerWeights.size(1));
		
	}
	
	/**
	 * This method implements the algorithm for deltabardelta but requires a gradient 
	 * from gradient collector class.
	 * @param gradientForHidden - gradient for hidden layer.
	 * @param gradientForOutput - gradient for output layer.
	 */
	public void Calculation(INDArray gradientForHidden, INDArray gradientForOutput)
	{
		for(int i=0; i<gradientForHidden.size(0); i++)
		{
			for(int j=0; j<gradientForHidden.size(1); j++)
			{
				
				if((gradientForHidden.getRow(i).getDouble(j) < 0.00 && previousGradientH.getRow(i).getDouble(j) < 0.00) || (gradientForHidden.getRow(i).getDouble(j) > 0 && previousGradientH.getRow(i).getDouble(j) > 0.00))
				{
					learningRateH.getRow(i).getColumn(j).assign(learningRateH.getRow(i).getDouble(j) + kGrowth);
				}else if((gradientForHidden.getRow(i).getDouble(j) > 0.00 && previousGradientH.getRow(i).getDouble(j) < 0.00) || (gradientForHidden.getRow(i).getDouble(j) > 0.00 && previousGradientH.getRow(i).getDouble(j) < 0.00)){
					
					learningRateH.getRow(i).getColumn(j).assign(learningRateH.getRow(i).getDouble(j) * (1 - dDecay));
				}
				
				gradientForHidden.getRow(i).getColumn(j).assign(gradientForHidden.getRow(i).getDouble(j) * learningRateH.getRow(i).getDouble(j));
				this.hiddenLayerWeights.getRow(i).getColumn(j).assign(this.hiddenLayerWeights.getRow(i).getDouble(j) - gradientForHidden.getRow(i).getDouble(j));
				
			}
		}
		
		for(int i=0; i<gradientForOutput.size(0); i++)
		{
			for(int j=0; j<gradientForOutput.size(1); j++)
			{
				
				if((gradientForOutput.getRow(i).getDouble(j) < 0.00 && previousGradientO.getRow(i).getDouble(j) < 0.00) || (gradientForOutput.getRow(i).getDouble(j) > 0 && previousGradientO.getRow(i).getDouble(j) > 0.00))
				{
					learningRateO.getRow(i).getColumn(j).assign(learningRateO.getRow(i).getDouble(j) + kGrowth);
				}else if((gradientForOutput.getRow(i).getDouble(j) > 0.00 && previousGradientO.getRow(i).getDouble(j) < 0.00) || (gradientForOutput.getRow(i).getDouble(j) > 0.00 && previousGradientO.getRow(i).getDouble(j) < 0.00)){
					
					learningRateO.getRow(i).getColumn(j).assign(learningRateO.getRow(i).getDouble(j) * (1 - dDecay));
				}
				
				gradientForOutput.getRow(i).getColumn(j).assign(gradientForOutput.getRow(i).getDouble(j) * learningRateH.getRow(i).getDouble(j));
				this.outputLayerWeights.getRow(i).getColumn(j).assign(this.outputLayerWeights.getRow(i).getDouble(j) - gradientForOutput.getRow(i).getDouble(j));
				
			}
		}
		
		previousGradientH = gradientForHidden;
		previousGradientO = gradientForOutput;

	}
}
