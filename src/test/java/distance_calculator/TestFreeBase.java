package distance_calculator;

import java.io.File;
import java.io.IOException;

import javax.ws.rs.DefaultValue;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;

import org.codehaus.jettison.json.JSONArray;
import org.codehaus.jettison.json.JSONException;
import org.codehaus.jettison.json.JSONObject;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.FeatureUtil;

public class TestFreeBase 
{
	
	public static void main( String[] args ) throws IOException {
		

		long startTime = System.currentTimeMillis();
//		WordVectors vec = WordVectorSerializer.loadTxtVectors(new File("C:/Users/yupeng/Downloads/knowledge-vectors-skipgram1000.txt"));
//		WordVectors vec = WordVectorSerializer.loadTxtVectors(new File("C:/Users/yupeng/Downloads/knowledge-vectors-skipgram1000-d4j-simple.txt"));
//		WordVectors vec = WordVectorSerializer.loadGoogleModel(new File("C:/Users/yupeng/Downloads/knowledge-vectors-skipgram1000.bin"), true);
		WordVectors vec = WordVectorSerializer.loadGoogleModel(new File("C:/Users/yupeng/Downloads/knowledge-vectors-skipgram1000.bin"), true);
		System.out.println("Loading completed in " + (System.currentTimeMillis() - startTime));
		System.out.println("Array size: " + vec.lookupTable().layerSize());
		
//		String source_word = "/m/048vr";
//		String source_word = "chinese";
		String source_word = "/m/05p553";


//		String[] target_words = new String[]{"/m/048vr","/m/01xw9","/m/07hxn","/m/01z1jf2","/m/051zk"};
//		String[] target_words = new String[]{"chinese","korean","japanese","thai","french","english","american","mexican"};
		String[] target_words = new String[]{"/m/05p553","/m/02822","/m/0217c8","/m/0jtdp","/m/082gq"};

		// Get the vector for the source and type word
		INDArray sourceVector = vec.getWordVectorMatrix(source_word);
		System.out.println("Source vector: " + source_word + ": "+ sourceVector.toString() + " [" +sourceVector.rows()+","+sourceVector.columns()+"]");
		
		System.out.println("Evaluate model....");
		
		for (String target_word : target_words){
			
			System.out.println("\n************************************************************************");

			if (vec.hasWord(target_word)){			
				INDArray targetVector = vec.getWordVectorMatrix(target_word);			
				
				System.out.println("Target vector: " + target_word + ": "+ targetVector.toString());

				double similarity = Transforms.cosineSim(targetVector,sourceVector);
				System.out.println("Similarity between "+ source_word + " vs. " + target_word + ":\t" + "(" + similarity + ")");
			} else {
				System.out.println(target_word + " is unknown.");
			}
			
			System.out.println("************************************************************************");
		}

	}
}
