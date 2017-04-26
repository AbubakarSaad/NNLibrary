package Library.learningrules;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import Library.functions.Functions;

public class Backprop {

	private INDArray outputSigmoidedValues;
	private INDArray errorContrAtOutput;
	Functions func = new Functions();
	private INDArray hiddenLayer;
	private INDArray outputLayer;
	
	public Backprop()
	{
		
	}
	
	public Backprop(INDArray hiddenLayer, INDArray outputLayer)
	 {
		this.hiddenLayer = hiddenLayer;
		this.outputLayer = outputLayer;
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
		 INDArray errorContrAtHidden = errorContrAtOutput.mmul(outputLayer.transpose());
		 
		 INDArray bi = func.sigmoid(hiddenLayerOutput, true);
		 INDArray errorAtHidden = errorContrAtHidden.muli(bi);
		 
		 
		 errorAtHidden = errorAtHidden.transpose();
		 
		 INDArray gradientForHidden = errorAtHidden.mmul(sample);
		 gradientForHidden = gradientForHidden.transpose();
		 
		 System.out.println("gradient for hidden: \n" + gradientForHidden);
		 
		 
	 }
	 
	 public INDArray exceptedOutput(int id)
		{
			if(Math.floor(id / 700) == 0)
			  {
			   return Nd4j.create(new double[]{1,0,0,0,0,0,0,0,0,0});
			  }else if(Math.floor(id / 700) == 1)
			  {
			   return Nd4j.create(new double[]{0,1,0,0,0,0,0,0,0,0});
			  }else if(Math.floor(id / 700) == 2)
			  {
			   return Nd4j.create(new double[]{0,0,1,0,0,0,0,0,0,0});
			  }else if(Math.floor(id / 700) == 3)
			  {
			   return Nd4j.create(new double[]{0,0,0,1,0,0,0,0,0,0});
			  }else if(Math.floor(id / 700) == 4)
			  {
			   return Nd4j.create(new double[]{0,0,0,0,1,0,0,0,0,0});
			  }else if(Math.floor(id / 700) == 5)
			  {
			   return Nd4j.create(new double[]{0,0,0,0,0,1,0,0,0,0});
			  }else if(Math.floor(id / 700) == 6)
			  {
			   return Nd4j.create(new double[]{0,0,0,0,0,0,1,0,0,0});
			  }else if(Math.floor(id / 700) == 7)
			  {
			   return Nd4j.create(new double[]{0,0,0,0,0,0,0,1,0,0});
			  }else if(Math.floor(id / 700) == 8)
			  {
			   return Nd4j.create(new double[]{0,0,0,0,0,0,0,0,1,0});
			  }else if(Math.floor(id / 700) == 9)
			  {
			   return Nd4j.create(new double[]{0,0,0,0,0,0,0,0,0,1});
			  }
			return null;
		}
}
