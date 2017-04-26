
package Library.generalizationtechniques;
import Library.learningrules.FeedForward;
import java.util.ArrayList;
import java.util.Collections;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.List;


public class TrainingTechniques {
	private List<INDArray> trainingData;
	private FeedForward ff;
	
	public TrainingTechniques(List<INDArray> trainingData, INDArray hiddenLayer, INDArray outputLayer){
		this.trainingData = trainingData;
		//trainingIndexArray();
		ff =  new FeedForward(hiddenLayer, outputLayer);
	}
	
	public void Holdout(int epochs){
		// Number of epochs is controlled by the user.
		for(int i = 0; i < epochs; i++){
			ArrayList<Integer> randomIndex = trainingIndexArray();
			for(int j = 0; j < trainingData.size(); j++){
				ff.forwardPass(trainingData.get(randomIndex.get(i)));
			}
		}
	}
	/**
	 * 
	 * @return - returns an 
	 */
	public ArrayList<Integer> trainingIndexArray(){
		ArrayList<Integer> randomIndex = new ArrayList<Integer>(trainingData.size());
		for (int i = 0; i <= trainingData.size(); i++){
            randomIndex.add(i);
		}
		Collections.shuffle(randomIndex);
		return randomIndex;
	}
	
	public double calcIndex(double index){
		double number = 0;
		number = index / 700;
		number = Math.floor(number);
		return number;
	}
}
