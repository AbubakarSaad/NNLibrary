package learningrul;

import org.nd4j.linalg.api.ndarray.INDArray;

public class FeedForward {

	private INDArray hiddenlayer;
	private INDArray outputlayer;
	
	public FeedForward(INDArray hiddenLayer, INDArray outputLayer)
	{
		hiddenlayer = hiddenLayer;
		outputlayer = outputLayer;
	}
	
	/**
	 * 
	 * @param row
	 */
	public void caluclations(INDArray row)
	{
		System.out.println(hiddenlayer);
		INDArray dotproduct = row.mmul(hiddenlayer);
		System.out.println("something works here");
		System.out.println(dotproduct);
	}
}
