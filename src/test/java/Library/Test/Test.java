package Library.Test;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import Library.neuron.Neuron;

/**
 * Created by Abu on 4/23/2017.
 */
public class Test {
    public Test()
    {
        INDArray nd = Nd4j.create(new float[]{1,2}, new int[]{2}); // row vectors
        INDArray nd2 = Nd4j.create(new float[]{3,4,5,6}, new int[]{2,2}); //column vector 
        
        System.out.println(nd);
        System.out.println(nd2);
        
        System.out.println("----answer---");
        System.out.println(nd.mmul(nd2));
        
        List<INDArray> list = new ArrayList<INDArray>();
        list.add(new Neuron(2, 3).getWeights());
        list.add(new Neuron(2, 3).getWeights());
        INDArray moreHiddenLayers = Nd4j.create(list, new int[]{2,2});
        System.out.println(list);
		System.out.println(moreHiddenLayers);
    }
    public static void main(String[] args)
    {
        new Test();
    }
}
