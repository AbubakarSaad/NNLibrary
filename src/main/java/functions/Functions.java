package functions;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Functions {

	public Functions()
	{
		
	}
	
	/**
	 * This is method for sigmoid
	 * @param input - takes an array to apply sigmoid
	 * @param x - true if the dreviate of sigmoid required
	 * @return - simgmoided values
	 */
	public INDArray sigmoid(INDArray input, Boolean x)
	{
		if(x == true)
		{
			return input.mul(input.sub(1, input));
			
		}
		return Transforms.sigmoid(input);
	}
	
	public INDArray exceptedOutput(int id)
	{
		return null;
	}
}
