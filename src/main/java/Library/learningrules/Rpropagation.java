
package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *This class contains the main implementation of rProp which works in 
 * conjunction with the gradient collector class.
 * @author Sulman and Abubakar
 */
public class Rpropagation {
	
	private double nNeg;
	private double nPos;
	private INDArray previousGradientH;
	private INDArray previousGradientO;
	private INDArray deltaWeightH;
	private INDArray deltaWeightO;
	private INDArray hiddenLayerWeights;
	private INDArray outputLayerWeights;
	
	
	public Rpropagation(double nNeg, double nPos, INDArray hiddenLayerWeights, INDArray outputLayerWeights)
	{
		this.nNeg = nNeg;
		this.nPos = nPos;
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		previousGradientH = Nd4j.zeros(hiddenLayerWeights.size(0), hiddenLayerWeights.size(1));
		previousGradientO = Nd4j.zeros(outputLayerWeights.size(0), outputLayerWeights.size(1));
		deltaWeightH = Nd4j.ones(hiddenLayerWeights.size(0), hiddenLayerWeights.size(1));
		deltaWeightO = Nd4j.ones(outputLayerWeights.size(0), outputLayerWeights.size(1));
	}
	
	/**
	 * This method that performs calculations for rProp.
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
					deltaWeightH.getRow(i).getColumn(j).assign(deltaWeightH.getRow(i).getDouble(j) * nPos);
				}else if((gradientForHidden.getRow(i).getDouble(j) > 0.00 && previousGradientH.getRow(i).getDouble(j) < 0.00) || (gradientForHidden.getRow(i).getDouble(j) > 0.00 && previousGradientH.getRow(i).getDouble(j) < 0.00)){
					
					deltaWeightH.getRow(i).getColumn(j).assign(deltaWeightH.getRow(i).getDouble(j) * nNeg);
				}else {
					gradientForHidden.getRow(i).getColumn(j).assign(gradientForHidden.getRow(i).getDouble(j));
				}
				
				if(gradientForHidden.getRow(i).getDouble(j) > 0.00)
				{
					this.hiddenLayerWeights.getRow(i).getColumn(j).assign(this.hiddenLayerWeights.getRow(i).getDouble(j) - gradientForHidden.getRow(i).getDouble(j));
				}else if(gradientForHidden.getRow(i).getDouble(j) < 0.00)
				{
					this.hiddenLayerWeights.getRow(i).getColumn(j).assign(this.hiddenLayerWeights.getRow(i).getDouble(j) + gradientForHidden.getRow(i).getDouble(j));
				}
			}
		}
		
		for(int i=0; i<gradientForOutput.size(0); i++)
		{
			for(int j=0; j<gradientForOutput.size(1); j++)
			{
				
				if((gradientForOutput.getRow(i).getDouble(j) < 0.00 && previousGradientO.getRow(i).getDouble(j) < 0.00) || (gradientForOutput.getRow(i).getDouble(j) > 0 && previousGradientO.getRow(i).getDouble(j) > 0.00))
				{
					deltaWeightO.getRow(i).getColumn(j).assign(deltaWeightO.getRow(i).getDouble(j) * nPos);
				}else if((gradientForOutput.getRow(i).getDouble(j) > 0.00 && previousGradientO.getRow(i).getDouble(j) < 0.00) || (gradientForOutput.getRow(i).getDouble(j) > 0.00 && previousGradientO.getRow(i).getDouble(j) < 0.00)){
					
					deltaWeightO.getRow(i).getColumn(j).assign(deltaWeightO.getRow(i).getDouble(j) * nNeg);
				}else {
					gradientForOutput.getRow(i).getColumn(j).assign(gradientForOutput.getRow(i).getDouble(j));
				}
				
				if(gradientForOutput.getRow(i).getDouble(j) > 0.00)
				{
					this.outputLayerWeights.getRow(i).getColumn(j).assign(this.outputLayerWeights.getRow(i).getDouble(j) - gradientForOutput.getRow(i).getDouble(j));
				}else if(gradientForOutput.getRow(i).getDouble(j) < 0.00)
				{
					this.outputLayerWeights.getRow(i).getColumn(j).assign(this.outputLayerWeights.getRow(i).getDouble(j) + gradientForOutput.getRow(i).getDouble(j));
				}
				
				
			}
		}
		
		previousGradientH = gradientForHidden;
		previousGradientO = gradientForOutput;
	}
}
