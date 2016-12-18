package distance_calculator;

import java.io.IOException;

public class Test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			Initialization.loadWord2VecModel("E:/Downloads/freebase-vectors-skipgram1000.bin/knowledge-vectors-skipgram1000.bin");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
