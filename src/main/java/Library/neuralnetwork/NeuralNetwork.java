package Library.neuralnetwork;
import Library.generalizationtechniques.TrainingTechniques;
import Library.layer.LayerCreation;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class NeuralNetwork {
	final String dir = System.getProperty("user.dir");
	private int epochs = 0;
	private int inputLayerSize = 64;
	private int hiddenLayerSize = 0;
	private int outputLayerSize = 0;
	private INDArray hiddenLayerWeights = null;
	private INDArray outputLayerWeights = null;
	private TrainingTechniques tt;
	
	public NeuralNetwork(int hiddenLayerSize, int outputLayerSize, int epochs){
		this.epochs = epochs;
		this.hiddenLayerWeights = createHiddenLayer(inputLayerSize, hiddenLayerSize);
		this.outputLayerWeights = createOutputLayer(hiddenLayerSize, outputLayerSize);
		
		// Intialize and sends training data and values at the hidden and output layer to trainingtechniques class.
		List<INDArray> trainingData = loadTrainingFiles(",");
		tt = new TrainingTechniques(trainingData, hiddenLayerWeights, outputLayerWeights);
		
	}
	
	public void holdoutTraining(){
		tt.Holdout(epochs);
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
	 * This method returns a list of type INDArray containing all the lines
	 * of every training file.
	 * @param delimeter - usually a comma that would be removed from the currentFile.
	 */
    public List<INDArray> loadTrainingFiles(String delimeter){
		List<String> allTrainingFiles = new ArrayList<String>();
		/** Add a variables for the folder in order to make it dynamic**/
		allTrainingFiles = loadAllFiles(dir + "//a1digits");
		List<INDArray> trainingData = new ArrayList<INDArray>();
		Iterator<String> iter = allTrainingFiles.iterator();
		
		//iterator used to iterate over the trainingData of files in allTrainingFiles and remove all test currentFile names.
		while(iter.hasNext()){
			if(iter.next().contains("test"))
				iter.remove();
		}
		
		INDArray currentFile = null;
		for(int i = 0; i < allTrainingFiles.size(); i++){
			try {
				currentFile = Nd4j.readNumpy(allTrainingFiles.get(i), delimeter);
				
			} catch (IOException ex) {
				Logger.getLogger(NeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
			}	
			for(int j = 0; j < currentFile.size(0); j++){
				trainingData.add(currentFile.getRow(j));
			}
		}
		return trainingData;
	}
	/**
	 * This method returns a list of type INDArray containing all the lines
	 * of every testing file.
	 * @param delimeter - usually a comma that would be removed from the currentFile.
	 */
    public List<INDArray> loadTestingFiles(String delimeter){
		List<String> allTestingFiles = new ArrayList<String>();
		allTestingFiles = loadAllFiles(dir + "//a1digits");
		List<INDArray> testData = new ArrayList<INDArray>();
		Iterator<String> iter = allTestingFiles.iterator();
		
		//iterator used to iterate over the testData of files in allTestingFiles and remove all training currentFile names.
		while(iter.hasNext()){
			if(iter.next().contains("training"))
				iter.remove();
		}
		
		INDArray currentFile = null;
		for(int i = 0; i < allTestingFiles.size(); i++){
			try {
				currentFile = Nd4j.readNumpy(allTestingFiles.get(i), delimeter);
				
			} catch (IOException ex) {
				Logger.getLogger(NeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
			}	
			for(int j = 0; j < currentFile.size(0); j++){
				testData.add(currentFile.getRow(j));
			}
		}
		return testData;
	}
	
    /**
     * 
     * @param inputlayerSize - size of the input layer 
     * @param numOfHiddenNeurons
     * @return
     */
	public INDArray createHiddenLayer(int numOfInputNeurons, int numOfHiddenNeurons)
	{
		
		return new LayerCreation(numOfInputNeurons, numOfHiddenNeurons).getWeights();
	}
	
	/**
	 * 
	 * @param numOfHiddenNeurons
	 * @param numOfOutputNeurons
	 * @return
	 */
	public INDArray createOutputLayer(int numOfHiddenNeurons, int numOfOutputNeurons)
	{
		return new LayerCreation(numOfHiddenNeurons, numOfOutputNeurons).getWeights();
	}
}
