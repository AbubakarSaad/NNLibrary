package Library.functions;

import org.nd4j.linalg.api.ndarray.INDArray;
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
	public INDArray sigmoid(INDArray inputx, Boolean x)
	{
		if(x == true)
		{
			return inputx.mul(inputx.sub(1));
		}
		return Transforms.sigmoid(inputx);
	}
	
	
}
