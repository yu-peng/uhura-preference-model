package distance_calculator;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class TestGoogleNews {


	public static void main( String[] args ) throws IOException
	{
		//    	Nd4j.getRandom().setSeed(133);
		//    	
		//    	System.err.println("Load data....");
		//        ClassPathResource resource = new ClassPathResource("raw_sentences.txt");
		//        SentenceIterator iter = new LineSentenceIterator(resource.getFile());
		//        iter.setPreProcessor(new SentencePreProcessor() {
		//            @Override
		//            public String preProcess(String sentence) {
		//                return sentence.toLowerCase();
		//            }
		//        });
		//        
		//        System.err.println("Tokenize data....");
		//        final EndingPreProcessor preProcessor = new EndingPreProcessor();
		//        TokenizerFactory tokenizer = new DefaultTokenizerFactory();
		//        tokenizer.setTokenPreProcessor(new TokenPreProcess() {
		//            @Override
		//            public String preProcess(String token) {
		//                token = token.toLowerCase();
		//                String base = preProcessor.preProcess(token);
		//                base = base.replaceAll("\\d", "d");
		//                if (base.endsWith("ly") || base.endsWith("ing"))
		//                    System.out.println();
		//                return base;
		//            }
		//        });
		//    	
		//        // Customizing params
		//        int batchSize = 1000;
		//        int iterations = 1;
		//        int layerSize = 300;
		//        
		//        System.err.println("Build model....");
		//        Word2Vec vec = new Word2Vec.Builder()
		//                .batchSize(batchSize)
		//                .sampling(1e-5)
		//                .minWordFrequency(5)
		//                .useAdaGrad(false)
		//                .layerSize(layerSize)
		//                .iterations(iterations)
		//                .learningRate(0.025)
		//                .minLearningRate(1e-2)
		//                .negativeSample(0)
		//                .iterate(iter)
		//                .tokenizerFactory(tokenizer)
		//                .build();
		//        vec.fit();
		//        
		//        System.err.println("Save vectors....");
		//        WordVectorSerializer.writeWordVectors(vec, "words.txt");

		//        InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();
		//        table.getSyn0().diviRowVector(table.getSyn0().norm2(0));

		long startTime = System.currentTimeMillis();
		
		WordVectors vec = WordVectorSerializer.loadTxtVectors(new File("D:/Downloads/glove.6B.100d.txt.gz"));

//		WordVectors vec = WordVectorSerializer.loadTxtVectors(new File("C:/Users/yupeng/Downloads/word2vec_models/glove.6B.100d.txt.gz"));
//		WordVectors vec = WordVectorSerializer.loadTxtVectors(new File("C:/Users/yupeng/Downloads/word2vec_models/glove.42B.300d.txt.gz"));
//		WordVectors vec = WordVectorSerializer.loadGoogleModel(new File("C:/Users/yupeng/Downloads/word2vec_models/GoogleNews-vectors-negative300.bin"), true);
//		WordVectors vec = WordVectorSerializer.loadGoogleModel(new File("C:/Users/yupeng/Downloads/word2vec_models/knowledge-vectors-skipgram1000.bin"), true);

//		InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();
//		table.getSyn0().diviRowVector(table.getSyn0().norm2(0));

		System.out.println("Loading completed in " + (System.currentTimeMillis() - startTime));

//		String source_word = "drama";
//		String type_word = "movie";
//		String[] target_words = new String[]{"action", "adventure","comedy","drama","horror","war","fiction","romance"};
		
		String source_word = "chinese";
		String type_word = "food";
		String[] target_words = new String[]{"chinese","korean","japanese","thai","french","english","american","mexican"};
				
		// Get the vector for the source and type word
		INDArray sourceVector = vec.getWordVectorMatrix(source_word);		
		INDArray typeVector = Nd4j.create(sourceVector.rows(),sourceVector.columns()); 		

		if (!type_word.isEmpty()){
			typeVector = vec.getWordVectorMatrix(type_word);	
		}

		System.out.println("Source vector: " + source_word + "/"+ sourceVector.toString());
		
		double type_sum = 0;
		
		for (int i = 0; i < typeVector.columns(); i++){
			type_sum += typeVector.getDouble(i)*typeVector.getDouble(i);
		}
		
		double type_average = type_sum/typeVector.columns();
		
		System.out.println("Type vector: " + type_word + "/"+ typeVector.toString());
		System.out.println("Type vector sum: " + type_word + "/"+ type_sum);
		System.out.println("Type vector average: " + type_word + "/"+ type_average);

		for (int i = 0; i < typeVector.columns(); i++){
			if(typeVector.getDouble(i)*typeVector.getDouble(i) > type_average){
				if (typeVector.getDouble(i) > 0){
					typeVector.putScalar(i, 1);
				} else {
					typeVector.putScalar(i, -1);
				}
			} else {
				typeVector.putScalar(i, 0);
			}
		}

		System.out.println("Type vector: " + type_word + "/"+ typeVector.toString());

		// Compute the difference from the source to the type vector
//		INDArray baseVector = sourceVector.add(typeVector);
		INDArray baseVector = sourceVector.mul(typeVector);
		System.out.println("Base vector: " + baseVector.toString());
		
		
		System.err.println("Evaluate model....");
		
		for (String target_word : target_words){
			INDArray targetVector = vec.getWordVectorMatrix(target_word);
//			INDArray targetRefVector = targetVector.add(typeVector);
			INDArray targetRefVector = targetVector.mul(typeVector);
			double similarity = Transforms.cosineSim(baseVector,targetRefVector);
			System.out.println("Similarity between "+ source_word + " vs. " + target_word + ":\t" + "(" + similarity + ")");
		}

	}

}
