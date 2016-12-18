package distance_calculator;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.HashSet;
import java.util.zip.GZIPInputStream;

import org.apache.commons.compress.compressors.gzip.GzipUtils;

public class ModelConverter {
	
    private static final int MAX_SIZE = 50;

	public static void main(String[] args) throws IOException {
		
		File modelFile = new File("C:/Users/yupeng/Downloads/knowledge-vectors-skipgram1000.bin"); 
		File outModelFile = new File("C:/Users/yupeng/Downloads/knowledge-vectors-skipgram1000.txt"); 

//		BufferedReader lineCounter = new BufferedReader(new FileReader("C:/Users/yupeng/Downloads/knowledge-vectors-skipgram1000.txt"));
//		BufferedWriter writer = new BufferedWriter(new FileWriter("C:/Users/yupeng/Downloads/knowledge-vectors-skipgram1000-d4j-simple.txt", false));
//
		HashSet<String> target_words = new HashSet<String>();
		target_words.add("/m/01xw9");
		target_words.add("/m/048vr");
		target_words.add("/m/042ck");
		target_words.add("/m/07hxn");
		target_words.add("/m/02z3r");
		target_words.add("/m/07s464v");
		target_words.add("/m/01z1jf2");
		target_words.add("/m/051zk");
//		
//        int lines = 0;
//        String content = lineCounter.readLine();
//        while (content != null) {
//        	
//        	if (lines == 0){
//        		System.out.println("First line: " + content);
//        	} else {
//        		String[] content_array = content.split(" ");
//        		if (target_words.contains(content_array[0])){
//                	writer.write(content+"\n");
//            		System.out.println(content_array[0] + ": " + content);
//        		}        		
//        	}
//        	
//        	content = lineCounter.readLine();
//        	
//        	lines++;
//        }
//        lineCounter.close();
//        writer.close();
//        
//        System.out.println("Lines in file: " + lines);
		
        BufferedWriter writer = new BufferedWriter(new FileWriter(outModelFile, false));

        int words, size;
        float vector;

		try (BufferedInputStream bis =
                new BufferedInputStream(GzipUtils.isCompressedFilename(modelFile.getName()) ?
                        new GZIPInputStream(new FileInputStream(modelFile)) :
                        new FileInputStream(modelFile));
                DataInputStream dis = new DataInputStream(bis)) {
			

            words = Integer.parseInt(readString(dis));
            size = Integer.parseInt(readString(dis));
            DecimalFormat df = new DecimalFormat("#.######"); 
            String word;
            for (int i = 0; i < words; i++) {

                word = readString(dis);
                if (word.isEmpty()) {
                    continue;
                }
                
                if (i % 1000 == 0){
                    System.out.print("Converting " + word + " of index " + i);
                }
                
                float[] wordVector = new float[size];
                float square_sum = 0;
                		
                for (int j = 0; j < size; j++) {
                    vector = dis.readFloat()*1.0e20f*1.0e20f;                      
                    wordVector[j] = vector;
                    square_sum += vector*vector;                    
                }
                
                double vector_length = Math.sqrt(square_sum);
                if (i % 1000 == 0){
                    System.out.print(" SQsum " + square_sum);
                    System.out.print(" length " + vector_length);
                    System.out.print("\n");
                }
                
                StringBuilder sb = new StringBuilder();
                sb.append(word.replaceAll(" ", "_"));
                sb.append(" ");
                
                for (int j = 0; j < size; j++) {
	                sb.append(df.format(wordVector[j]/vector_length));
	                if (j < size - 1) {
	                    sb.append(" ");
	                }
                }
                sb.append("\n");
                writer.write(sb.toString());
            }
        }		

        writer.flush();
        writer.close();
	}
	
	/**
     * Read a string from a data input stream Credit to:
     * https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
     *
     * @param dis
     * @return
     * @throws IOException
     */
    private static String readString(DataInputStream dis)
        throws IOException
    {
        byte[] bytes = new byte[MAX_SIZE];
        byte b = dis.readByte();
        int i = -1;
        StringBuilder sb = new StringBuilder();
        while (b != 32 && b != 10) {
            i++;
            bytes[i] = b;
            b = dis.readByte();
            if (i == 49) {
                sb.append(new String(bytes));
                i = -1;
                bytes = new byte[MAX_SIZE];
            }
        }
        sb.append(new String(bytes, 0, i + 1));
        return sb.toString();
    }

}
