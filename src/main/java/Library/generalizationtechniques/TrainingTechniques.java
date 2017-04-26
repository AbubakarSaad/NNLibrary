
package Library.generalizationtechniques;
import Library.learningrules.Backprop;
import Library.learningrules.FeedForward;
import java.util.ArrayList;
import java.util.Collections;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import org.nd4j.linalg.factory.Nd4j;


public class TrainingTechniques {
	private List<INDArray> trainingData;
	private FeedForward ff;
	private Backprop bp;
	private INDArray hiddenLayerWeights;
	private INDArray outputLayerWeights;
	private double correctAnswer = 0;
	
	public TrainingTechniques(List<INDArray> trainingData, INDArray hiddenLayerWeights, INDArray outputLayerWeights){
		this.hiddenLayerWeights = hiddenLayerWeights;
		this.outputLayerWeights = outputLayerWeights;
		this.trainingData = trainingData;
		
		ff =  new FeedForward(hiddenLayerWeights, outputLayerWeights);
		bp = new Backprop(hiddenLayerWeights, outputLayerWeights);
	}
	
	public void Holdout(int epochs){
		// Number of epochs is controlled by the user.
		
		for(int i = 0; i < epochs; i++){
			ArrayList<Integer> randomIndex = trainingIndexArray();
			correctAnswer = 0;
			for(int j = 0; j < trainingData.size(); j++){
				ff.forwardPass(trainingData.get(randomIndex.get(j)));
				INDArray errorAtOutput = ff.getOutputofOutputLayer().sub(expectedOutput(randomIndex.get(j)));
				accuracy(randomIndex.get(j));
				bp.calculations(ff.getOutputofOutputLayer(), ff.getOutputofHiddenLayer(), errorAtOutput, trainingData.get(randomIndex.get(j)));
				hiddenLayerWeights = bp.getUpdatedHiddenLayerWeights();
				outputLayerWeights = bp.getUpdatedOutputLayerWeights();
				
			}
			System.out.println("Epoch: " + i);
			System.out.println("Accuracy per Epoch: " + correctAnswer / trainingData.size());
			System.out.println("# correct: " + correctAnswer);
			//System.out.println("reset " + correctAnswer);
		}
	}
	/**
	 * 
	 * @return - returns an 
	 */
	public ArrayList<Integer> trainingIndexArray(){
		ArrayList<Integer> randomIndex = new ArrayList<Integer>(trainingData.size());
		for (int i = 0; i < trainingData.size(); i++){
            randomIndex.add(i);
		}
		Collections.shuffle(randomIndex);
		return randomIndex;
	}
	public void accuracy(int index){
		INDArray target = expectedOutput(index);
		INDArray outputLayerValues = ff.getOutputofOutputLayer();
		
		for(int i = 0; i <target.size(1); i++){
			if(target.getDouble(i) == 1){
				//System.out.println("Reached");
				//System.out.println(outputLayerValues.getDouble(i));
				if(outputLayerValues.getDouble(i) > 0.50){
					//System.out.println("Reached 2");
					//System.out.println("Max number in outputlayer: " + outputLayerValues.maxNumber());
					if(outputLayerValues.getDouble(i) == outputLayerValues.maxNumber().doubleValue()){
						//System.out.println("Target array:" + target);
						//System.out.println("Output values:" + outputLayerValues);
						correctAnswer++;
						//System.out.println("Correct answer: "+ correctAnswer);
						break;
					}
				}
			}
		}
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
