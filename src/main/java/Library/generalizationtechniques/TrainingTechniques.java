
package Library.generalizationtechniques;
import Library.learningrules.FeedForward;
import java.util.ArrayList;
import java.util.Collections;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
	
	public INDArray exceptedOutput(int id)
	{
		if(Math.floor(id / 700) == 0)
		  {
		   return Nd4j.create(new double[]{1,0,0,0,0,0,0,0,0,0});
		  }else if(Math.floor(id / 700) == 1)
		  {
		   return Nd4j.create(new double[]{0,1,0,0,0,0,0,0,0,0});
		  }else if(Math.floor(id / 700) == 2)
		  {
		   return Nd4j.create(new double[]{0,0,1,0,0,0,0,0,0,0});
		  }else if(Math.floor(id / 700) == 3)
		  {
		   return Nd4j.create(new double[]{0,0,0,1,0,0,0,0,0,0});
		  }else if(Math.floor(id / 700) == 4)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,1,0,0,0,0,0});
		  }else if(Math.floor(id / 700) == 5)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,0,1,0,0,0,0});
		  }else if(Math.floor(id / 700) == 6)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,0,0,1,0,0,0});
		  }else if(Math.floor(id / 700) == 7)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,0,0,0,1,0,0});
		  }else if(Math.floor(id / 700) == 8)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,0,0,0,0,1,0});
		  }else if(Math.floor(id / 700) == 9)
		  {
		   return Nd4j.create(new double[]{0,0,0,0,0,0,0,0,0,1});
		  }
		return null;
	}
}
