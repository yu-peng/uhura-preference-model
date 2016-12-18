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
import org.nd4j.linalg.ops.transforms.Transforms;

@Path("/Word2VecFreeBase")
public class W2VFreeBase 
{
	
	@GET
	@Produces(MediaType.APPLICATION_JSON)
	public String computeDistances(
			@DefaultValue("") @QueryParam("source") String source,
			@DefaultValue("") @QueryParam("targets") String targets) throws JSONException {
		
		JSONObject response = new JSONObject();
		response.put("source", source);
		response.put("targets", targets);

		if (source.isEmpty()){
			
			response.put("Error","Source word is not specified.");
			return response.toString();
			
		} else if (!Initialization.vec.hasWord(source)){
			
			response.put("Error","Source word is not found in the model.");
			return response.toString();
			
		} else if (targets.isEmpty()){
					
			response.put("Error","Target words are not specified.");
			return response.toString();

		}
		
		// Get the vector for the source and type word
		INDArray sourceVector = Initialization.vec.getWordVectorMatrix(source);		

		JSONArray result_collection = new JSONArray();

		String[] targetArray = targets.split(",");
		// Go through each of the target words
		for (String target : targetArray){
			
			JSONObject result = new JSONObject();
			result.put("word", target);
			
			if (Initialization.vec.hasWord(target)){
				
				INDArray targetVector = Initialization.vec.getWordVectorMatrix(target);
				double similarity = Transforms.cosineSim(targetVector,sourceVector);

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
