package Library.neuralnetwork;

import Library.layer.LayerCreation;
import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;


public class NeuralNetwork {
	final String dir = System.getProperty("user.dir");
	
	public NeuralNetwork(){
		
		loadTrainingFiles();
		// neuron(sizeofInput, numberofHidden)
		//INDArray hiddenLayer = getHiddenLayer(training_files.size(1), 2);
		//System.out.println(hiddenLayer);
		
		//INDArray outputLayer = getOutputLayer(hiddenLayer.size(1), 10);
	}
	
	public void holdoutTraining(){
		
	}
	
	public List<String> loadAllFiles(String path){
		List<String> allFiles = new ArrayList<String>();
		String directory = path;
		File[] files = new File(directory).listFiles();
		
		for(File file : files){
			if(file.isFile()){
				allFiles.add(file.getAbsolutePath());
			}
		}
		return allFiles;
	}
	
	/**
	 * 
	 * @param fileName
	 * @param delimiter
	 * @return
	 */
    public INDArray loadTrainingFiles(){

		List<String> allTrainingFiles = new ArrayList<String>();
		allTrainingFiles = loadAllFiles(dir + "//a1digits");
		Iterator<String> iter = allTrainingFiles.iterator();
		
        while(iter.hasNext()){
            if(iter.next().contains("test"))
                iter.remove();
		}
		System.out.println(allTrainingFiles);
//		try {
//			training_files = Nd4j.readNumpy(fileName, delimiter);
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
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
