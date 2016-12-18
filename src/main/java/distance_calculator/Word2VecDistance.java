package distance_calculator;

import javax.ws.rs.DefaultValue;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;

import org.codehaus.jettison.json.JSONArray;
import org.codehaus.jettison.json.JSONException;
import org.codehaus.jettison.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

@Path("/Word2VecSimilarity")
public class Word2VecDistance 
{
	
	@GET
	@Produces(MediaType.APPLICATION_JSON)
	public String computeDistances(
			@DefaultValue("") @QueryParam("source") String sourceWord,
			@DefaultValue("") @QueryParam("targets") String targetWords,
			@DefaultValue("") @QueryParam("type") String typeWord) throws JSONException {
		
		JSONObject response = new JSONObject();
		response.put("source", sourceWord);
		response.put("targets", targetWords);
		response.put("type", typeWord);

		if (sourceWord.isEmpty()){
			
			response.put("Error","Source word is not specified.");
			return response.toString();
			
		} else if (!Initialization.vec.hasWord(sourceWord)){
			
			response.put("Error","Source word is not found in the model.");
			return response.toString();
			
		} else if (!typeWord.isEmpty() && !Initialization.vec.hasWord(typeWord)){
			
			response.put("Error","Type word is not found in the model.");
			return response.toString();
			
		} else if (targetWords.isEmpty()){
					
			response.put("Error","Target words are not specified.");
			return response.toString();

		}
		
		// Get the vector for the source and type word
		INDArray sourceVector = Initialization.vec.getWordVectorMatrix(sourceWord);		
		INDArray typeVector = Nd4j.create(sourceVector.rows(),sourceVector.columns()); 		

		if (!typeWord.isEmpty()){
			typeVector = Initialization.vec.getWordVectorMatrix(typeWord);	
			
			double type_sum = 0;
			
			for (int i = 0; i < typeVector.columns(); i++){
				type_sum += typeVector.getDouble(i)*typeVector.getDouble(i);
			}
			
			double type_average = type_sum/typeVector.columns();

			for (int i = 0; i < typeVector.columns(); i++){
				if(typeVector.getDouble(i)*typeVector.getDouble(i) > type_average){
//					if (typeVector.getDouble(i) > 0){
						typeVector.putScalar(i, 1);
//					} else {
//						typeVector.putScalar(i, -1);
//					}
				} else {
					typeVector.putScalar(i, 0);
				}
			}
		} else {
			for (int i = 0; i < typeVector.columns(); i++){
				typeVector.putScalar(i, 1);
			}
		}
		
		// Compute the difference from the source to the type vector
//		INDArray baseRefVector = sourceVector.add(typeVector);
		INDArray baseRefVector = sourceVector.mul(typeVector);

		JSONArray result_collection = new JSONArray();

		String[] targetWordArray = targetWords.split(",");
		// Go through each of the target words
		for (String targetWord : targetWordArray){
			
			JSONObject result = new JSONObject();
			result.put("word", targetWord);
			
			if (Initialization.vec.hasWord(targetWord)){
				
				INDArray targetVector = Initialization.vec.getWordVectorMatrix(targetWord);
//				INDArray targetRefVector = targetVector.add(typeVector);
				INDArray targetRefVector = targetVector.mul(typeVector);
				double similarity = Transforms.cosineSim(targetRefVector,baseRefVector);

				result.put("similarity", similarity);
				
			} else {
				
//				result.put("similarity", -10);
				
			}
			
			result_collection.put(result);

		}
		
		response.put("results", result_collection);

		return response.toString();
	}
	
}
