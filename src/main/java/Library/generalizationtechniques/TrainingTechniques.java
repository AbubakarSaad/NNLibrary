
package Library.generalizationtechniques;
import org.nd4j.linalg.api.ndarray.INDArray;
import Library.learningrules.FeedForward;


public class TrainingTechniques {
	private INDArray trainingdata;
	
	public TrainingTechniques(INDArray trainingdata){
		this.trainingdata = trainingdata;
		//FeedForward ff = new FeedForward(hiddenLayer, outputLayer);
	}
	
	public void Holdout(int epochs){
		System.out.println("Training data length: " + this.trainingdata.size(0));
		
		// Number of epochs is controlled by user.
		for(int i = 0; i < epochs; i++){
			for(int j = 0; j < this.trainingdata.size(0); j++){
				
			}
		}
	}
}
