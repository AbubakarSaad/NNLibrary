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

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This is the main class that connects all other classes, this is what the 
 * user instantiates in the main class and calls whatever training method is
 * needed.
 * @author Sulman and Abubakar
 */
public class NeuralNetwork {
	final String dir = System.getProperty("user.dir");
	private int epochs = 0;
	private int inputLayerSize = 64;
	private int hiddenLayerSize = 0;
	private int outputLayerSize = 0;
	private INDArray hiddenLayerWeights = null;
	private INDArray outputLayerWeights = null;
	private TrainingTechniques tt;
	
	/**
	 * The first constructor which gives the option of putting a learning rate 
	 * in.
	 * @param hiddenLayerSize - specified by the user
	 * @param outputLayerSize - specified by the user
	 * @param epochs - specified by the user
	 * @param learningRate - specified by the user
	 */
	public NeuralNetwork(int hiddenLayerSize, int outputLayerSize, int epochs, double learningRate){
		this.epochs = epochs;
		//Randomly generate the weight connections for both hidden and output
		//layer
		this.hiddenLayerWeights = createHiddenLayer(inputLayerSize, hiddenLayerSize);
		this.outputLayerWeights = createOutputLayer(hiddenLayerSize, outputLayerSize);
		
		// Intialize and send training data and the values at the hidden and 
		//outputlayer to trainingtechniques class.
		List<INDArray> trainingData = loadTrainingFiles(",");
		tt = new TrainingTechniques(trainingData, this.hiddenLayerWeights, this.outputLayerWeights, learningRate);
		
	}
	/**
	 * This constructer is used when the user only wants bias or learning rate 
	 * implemented. To choose you simply type in the last parameter a string 
	 * either "bias" or "learning". 
	 * @param hiddenLayerSize
	 * @param outputLayerSize
	 * @param epochs
	 * @param learningRate
	 * @param learningParam 
	 */
	public NeuralNetwork(int hiddenLayerSize, int outputLayerSize, int epochs, double learningRate, String learningParam){
		this.epochs = epochs;
		this.hiddenLayerWeights = createHiddenLayer(inputLayerSize, hiddenLayerSize);
		this.outputLayerWeights = createOutputLayer(hiddenLayerSize, outputLayerSize);
		
		// Intialize and sends training data and values at the hidden and output layer to trainingtechniques class.
		List<INDArray> trainingData = loadTrainingFiles(",");
		tt = new TrainingTechniques(trainingData, hiddenLayerWeights, outputLayerWeights, learningRate, "bias");
		
	}
	/**
	 * This constructer is used when the user would like to use all learning 
	 * rules available in the library such as learningRate, bias and momentum.
	 * @param hiddenLayerSize - specified by the user
	 * @param outputLayerSize - specified by the user
	 * @param epochs - specified by the user
	 * @param learningRate - specified by the user
	 * @param bias - specified by the user
	 * @param momentum - specified by the user
	 */
	public NeuralNetwork(int hiddenLayerSize, int outputLayerSize, int epochs, double learningRate, String bias, String momentum){
		Nd4j.setDataType(DataBuffer.Type.DOUBLE);
		this.epochs = epochs;
		this.hiddenLayerWeights = createHiddenLayer(inputLayerSize, hiddenLayerSize);
		this.outputLayerWeights = createOutputLayer(hiddenLayerSize, outputLayerSize);
		
		// Intialize and sends training data and values at the hidden and output layer to trainingtechniques class.
		List<INDArray> trainingData = loadTrainingFiles(",");
		tt = new TrainingTechniques(trainingData, hiddenLayerWeights, outputLayerWeights, learningRate);
		
	}
	/**
	 * This method is what the user calls to run holdout training after the 
	 * network has been initialized.
	 */
	public void holdoutTraining(){
		tt.Holdout(epochs);
	}
	
	/**
	 * This method is what the user calls to run batch training with rprop
	 */
	public void rprop(){
		tt.batchTraining(epochs, 1.2, 0.5);
	}
	
	/**
	 * This method is what user calls to run batch training with delta_bar_delta
	 */
	public void deltabardelta()
	{
		tt.batchTrainingD(epochs, 0.20, 0.0001);
	}
	
	/**
	 * Loads all training and test file paths into a string list which is used 
	 * later in loadTrainingFiles and loadTestingfiles.
	 * @param path - the absolute path leading to the testing and training 
	 * files.
	 * @return - List<String> 
	 */
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
	 * @param delimeter - usually a comma that would be removed from the 
	 * currentFile.
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
     * This method creates the connection matrix between the input layer and 
	 * hidden layer.
     * @param inputlayerSize - size of the input layer which will be used as the 
	 * first dimension of the array.
     * @param numOfHiddenNeurons - size of the hidden layer that will be used
	 * for the second dimension of the array.
     * @return - returns a INDArray containing random weights varying from
	 * -0.5 to 0.5.
     */
	public INDArray createHiddenLayer(int numOfInputNeurons, int numOfHiddenNeurons)
	{
		return new LayerCreation(numOfInputNeurons, numOfHiddenNeurons).getWeights();
	}
	
	/**
	 * This method creates the connection matrix between the hidden layer and 
	 * output layer.
	 * @param numOfHiddenNeurons - size of the hidden layer which is used as the
	 * first dimension of the weight array.
	 * @param numOfOutputNeurons - size of the output layer which is used as the
	 * second dimension of the weight array.
	 * @return - returns a INDArray containing random weights varying from
	 * -0.5 to 0.5.
	 */
	public INDArray createOutputLayer(int numOfHiddenNeurons, int numOfOutputNeurons)
	{
		return new LayerCreation(numOfHiddenNeurons, numOfOutputNeurons).getWeights();
	}
}
