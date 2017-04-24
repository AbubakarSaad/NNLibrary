package functions;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Functions {

	public Functions()
	{
		
	}
	
	public INDArray sigmoid(INDArray input, Boolean x)
	{
		if(x == true)
		{
			return input.mul(input.sub(1, input));
			
		}
		return Transforms.sigmoid(input);
	}
}
