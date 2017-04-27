package Library.functions;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Functions {

	public Functions()
	{
		
	}
	
	/**
	 * This is method for sigmoid
	 * @param input - takes an array to apply sigmoid
	 * @param x - true if the derivative of sigmoid required
	 * @return - sigmoided values
	 */
	public INDArray sigmoid(INDArray input, Boolean x)
	{
		if(x == true)
		{
			return input.mul(input.rsub(1));

		}else {
			return Transforms.sigmoid(input);
		}
	}
	
	
}
