
package Library.generalizationtechniques;
import Library.learningrules.Backprop;
import Library.learningrules.FeedForward;
import java.util.ArrayList;
import java.util.Collections;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This class holds all the implementations of training techniques such as 
 * holdout or k-fold cross validation.
 * @author Sulman and Abubakar
 */
public class TrainingTechniques {
	private List<INDArray> trainingData;
	private FeedForward ff;
	private Backprop bp;
	private INDArray hiddenLayerWeights;
	private INDArray outputLayerWeights;

	private double correctAnswer = 0;

	private INDArray biasArrayH;
	private INDArray biasArrayO;

	
	public TrainingTechniques(List<INDArray> trainingData, INDArray hiddenLayerWeights, INDArray outputLayerWeights, double learningRate){
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		this.trainingData = trainingData;
		
		ff =  new FeedForward(hiddenLayerWeights, outputLayerWeights);
		bp = new Backprop(hiddenLayerWeights, outputLayerWeights, learningRate);
		//System.out.println("hiddenLayerweight began: \n"+hiddenLayerWeights);
	}
	
	/**
	 * This is the constructer that includes bias or momentum.
	 * @param trainingData - holds the training data.
	 * @param hiddenLayerWeights - holds the hidden layer weights values.
	 * @param outputLayerWeights - holds the output layer weights values.
	 * @param learningParams - ask for the bias or momentum included in the learning.
	 */
	public TrainingTechniques(List<INDArray> trainingData, INDArray hiddenLayerWeights, INDArray outputLayerWeights, double learningRate, String learningParams){
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		this.trainingData = trainingData;
		//trainingIndexArray();
		if (learningParams == "bias")
		{
			biasArrayH = Nd4j.rand(1, this.hiddenLayerWeights.size(1), -0.5, 0.5, Nd4j.getRandom());
			biasArrayO = Nd4j.rand(1, this.outputLayerWeights.size(1), -0.5, 0.5, Nd4j.getRandom());
		}
		ff =  new FeedForward(hiddenLayerWeights, outputLayerWeights, biasArrayH, biasArrayO);
		bp = new Backprop(hiddenLayerWeights, outputLayerWeights, learningRate);
	}
	/**
	 * The main implementation of the holdout method which is called in the 
	 * NeuralNetwork class, has an outer epoch loop which is controlled by the 
	 * user and run through the main algorithm of learning.
	 * @param epochs - selected by the user of the library.
	 */
	public void Holdout(int epochs){
		// Number of epochs is controlled by the user.
		
		for(int i = 0; i < epochs; i++){
			ArrayList<Integer> randomIndex = trainingIndexArray();
			correctAnswer = 0;
			for(int j = 0; j < trainingData.size()-6997; j++){
				ff.forwardPass(trainingData.get(randomIndex.get(j)));
				INDArray errorAtOutput = ff.getOutputofOutputLayer().sub(expectedOutput(randomIndex.get(j)));
				accuracy(randomIndex.get(j));
				bp.calculations(ff.getOutputofOutputLayer(), ff.getOutputofHiddenLayer(), errorAtOutput, trainingData.get(randomIndex.get(j)));
				hiddenLayerWeights = bp.getUpdatedHiddenLayerWeights();
				outputLayerWeights = bp.getUpdatedOutputLayerWeights();	
				
			}
			System.out.println("Epoch: " + i);
			System.out.println("Accuracy per Epoch: " + correctAnswer / trainingData.size());
		}
	}
	/**
	 * Creates a randomized list of integers based on the number of training
	 * examples.
	 * @return - returns an ArrayList<Integer> containing the ints.
	 */
	public ArrayList<Integer> trainingIndexArray(){
		ArrayList<Integer> randomIndex = new ArrayList<Integer>(trainingData.size());
		for (int i = 0; i < trainingData.size(); i++){
            randomIndex.add(i);
		}
		Collections.shuffle(randomIndex);
		return randomIndex;
	}
	/**
	 * Tracks the accuracy of the the current epoch using a counter.
	 * @param index 
	 */
	public void accuracy(int index){
		INDArray target = expectedOutput(index);
		INDArray outputLayerValues = ff.getOutputofOutputLayer();
		
		for(int i = 0; i <target.size(1); i++){
			if(target.getDouble(i) == 1){
				if(outputLayerValues.getDouble(i) > 0.50){
					if(outputLayerValues.getDouble(i) == outputLayerValues.maxNumber().doubleValue()){
						System.out.println("Target array:" + target);
						correctAnswer++;
						break;
					}
				}
			}
		}
	}	

	public void k_foldCrossValidation(int epochs, int k){
		for(int i = 0; i < epochs; i++){
			INDArray averageAccuracy = Nd4j.create(new double[k]);
			ArrayList<Integer> randomIndexes = trainingIndexArray();
			for(int j = 0; j < k; j++){
				
			}
		}
	}	
	/**
	 * This method calculates and returns the expected output for the given 
	 * data.
	 * @param index - the position of the data in the training file.
	 * @return - INDArray an expected output array.
	 */
	public INDArray expectedOutput(int index){
		if(Math.floor(index / 700) == 0)
		  {
		   return Nd4j.create(new double[]{1,0,0,0,0,0,0,0,0,0});
		  }else if(Math.floor(index / 700) == 1)
		  {
		   return Nd4j.create(new double[]{0,1,0,0,0,0,0,0,0,0});
		  }else if(Math.floor(index / 700) == 2)
		  {
		   return Nd4j.create(new double[]{0,0,1,0,0,0,0,0,0,0});
		  }else if(Math.floor(index / 700) == 3)
		  {
		   return Nd4j.create(new double[]{0,0,0,1,0,0,0,0,0,0});
		  }else if(Math.floor(index / 700) == 4)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,1,0,0,0,0,0});
		  }else if(Math.floor(index / 700) == 5)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,0,1,0,0,0,0});
		  }else if(Math.floor(index / 700) == 6)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,0,0,1,0,0,0});
		  }else if(Math.floor(index / 700) == 7)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,0,0,0,1,0,0});
		  }else if(Math.floor(index / 700) == 8)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,0,0,0,0,1,0});
		  }else if(Math.floor(index / 700) == 9)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,0,0,0,0,0,1});
		  }
		return null;
	}
	
}
