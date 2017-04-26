
package Library.generalizationtechniques;
import Library.learningrules.Backprop;
import Library.learningrules.FeedForward;
import java.util.ArrayList;
import java.util.Collections;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import org.nd4j.linalg.factory.Nd4j;


public class TrainingTechniques {
	private List<INDArray> trainingData;
	private FeedForward ff;
	private Backprop bp;
	private INDArray hiddenLayerWeights;
	private INDArray outputLayerWeights;
	
	public TrainingTechniques(List<INDArray> trainingData, INDArray hiddenLayerWeights, INDArray outputLayerWeights){
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		this.trainingData = trainingData;
		//trainingIndexArray();
		ff =  new FeedForward(hiddenLayerWeights, outputLayerWeights);
		bp = new Backprop(hiddenLayerWeights, outputLayerWeights);
		System.out.println("hiddenLayerweight began: \n"+hiddenLayerWeights);
	}
	
	public void Holdout(int epochs){
		// Number of epochs is controlled by the user.
		
		for(int i = 0; i < epochs; i++){
			ArrayList<Integer> randomIndex = trainingIndexArray();
			for(int j = 0; j < trainingData.size() - 6999; j++){
				ff.forwardPass(trainingData.get(randomIndex.get(i)));
				INDArray errorAtOutput = ff.getOutputofOutputLayer().sub(expectedOutput(randomIndex.get(i)));
				
				bp.calculations(ff.getOutputofOutputLayer(), ff.getOutputofHiddenLayer(), errorAtOutput, trainingData.get(randomIndex.get(i)));
				
				hiddenLayerWeights = bp.getUpdatedHiddenLayerWeights();
				outputLayerWeights = bp.getUpdatedOutputLayerWeights();
				
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
