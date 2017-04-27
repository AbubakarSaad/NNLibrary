package Library.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import Library.functions.Functions;
import Library.layer.LayerCreation;
import Library.learningrules.Backprop;
import Library.learningrules.FeedForward;

/**
 * Created by Abu on 4/23/2017.
 */
public class Test {
	
	final String dir = System.getProperty("user.dir");
	public int epoch = 1;
	
	Functions func = new Functions();
	private INDArray hiddenLayerOutput;
	private INDArray outputLayerOutput;
    public Test()
    {
    	//LayerCreation lc = new LayerCreation(10, 5);
    	//System.out.println(lc.getWeights());

    	INDArray tes = Nd4j.create(new double[] {0.1, 0.2});
    	INDArray tes2 = Nd4j.create(new double[]{0.3, 0.4});
    	System.out.println(tes + " " + tes2);
    	System.out.println(tes.sub(tes2));
		
    }
    
    public static void main(String[] args)
    {
        new Test();
    }
}
