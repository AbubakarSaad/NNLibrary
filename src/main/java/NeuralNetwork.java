
import Library.layer.LayerCreation;
import java.io.IOException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class NeuralNetwork {
	
	public NeuralNetwork(){
		final String dir = System.getProperty("user.dir");
		    	
		// input layer 
		INDArray inputLayer = getInputLayer(dir + "//a1digits//digit_test_0.txt", ",");
		//System.out.print(inputLayer);
		
		// neuron(sizeofInput, numberofHidden)
		INDArray hiddenLayer = getHiddenLayer(inputLayer.size(1), 2);
		//System.out.println(hiddenLayer);
		
		INDArray outputLayer = getOutputLayer(hiddenLayer.size(1), 10);
	}
	
	public void holdoutTraining(){
		
	}
	
	/**
	 * 
	 * @param fileName
	 * @param delimiter
	 * @return
	 */
    public INDArray getInputLayer(String fileName, String delimiter)
	{
		try {
			return Nd4j.readNumpy(fileName, delimiter);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
    /**
     * 
     * @param inputlayerSize - size of the input layer 
     * @param numOfHiddenNeurons
     * @return
     */
	public INDArray getHiddenLayer(int inputlayerSize, int numOfHiddenNeurons)
	{
		
		return new LayerCreation(inputlayerSize, numOfHiddenNeurons).getWeights();
	}
	
	/**
	 * 
	 * @param hiddenLayerSize
	 * @param numOfOutputNeurons
	 * @return
	 */
	public INDArray getOutputLayer(int hiddenLayerSize, int numOfOutputNeurons)
	{
		return new LayerCreation(hiddenLayerSize, numOfOutputNeurons).getWeights();
	}
}
