package distance_calculator;

import java.io.File;
import java.io.IOException;
import java.sql.Driver;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.servlet.ServletContextEvent;
import javax.servlet.ServletContextListener;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Functions that are required for initializing the
 * path planner.
 * 
 * @author Peng Yu
 *
 */
public class Initialization implements ServletContextListener{

	static String modelPathPrefix = null;
	public static WordVectors vec = null;
	
	public void contextInitialized(ServletContextEvent arg0) {
		modelPathPrefix = "/word2vec_models/";
		String model_path = modelPathPrefix + "knowledge-vectors-skipgram1000.bin";

		try {
			loadWord2VecModel(model_path);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void loadWord2VecModel(String model_path) throws IOException{
		
		vec = WordVectorSerializer.loadGoogleModel(new File(model_path), true);
		String[] mids = new String[]{"/m/01xw9","/m/048vr","/m/09y2k2","/m/01h5q0","/m/042ck","/m/02z3r","/m/01z1jf2","/m/04v5by","/m/051zk","/m/06nkpw","/m/07_19","/m/07hxn"};
		String[] cuisines = new String[]{"Chinese","Korean","Italian","Indian","Japanese","French","American","Mediterranean","Mexican","Iranian","Vietnamese","Thai"};

		ArrayList<String> found_mids = new ArrayList<String>();
		ArrayList<String> found_cuisines = new ArrayList<String>();
		
		System.out.print("X = [");	
		int idx = 0;
		for (String mid : mids){
			if (vec.hasWord(mid)){
				INDArray vector = Initialization.vec.getWordVectorMatrix(mid);
				
				for (int i = 0; i < vector.length(); i++){
					if (i < vector.length()-1){
						System.out.print(vector.getDouble(i)+",");
					} else {
						System.out.print(vector.getDouble(i));						
					}
				}
				
				System.out.print(";\n");
				found_mids.add(mid);
				found_cuisines.add(cuisines[idx]);
			}
			idx++;
		}
		System.out.print("];\n");
		
		System.out.print("labels = {'");
		System.out.print(String.join("','", found_cuisines));
		System.out.print("'};");
	}

	public void loadWord2TxtVecModel(String model_path) throws IOException{
		vec = WordVectorSerializer.loadTxtVectors(new File(model_path));
	}
	
	public void contextDestroyed(ServletContextEvent arg0) {

		Logger LOG = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
		
		Enumeration<Driver> drivers = DriverManager.getDrivers();
		while (drivers.hasMoreElements()) {
			Driver driver = drivers.nextElement();
			try {
				DriverManager.deregisterDriver(driver);
				LOG.log(Level.INFO, String.format("deregistering jdbc driver: %s", driver));
			} catch (SQLException e) {
				LOG.log(Level.SEVERE, String.format("Error deregistering driver %s", driver), e);
			}

		}

	}
}
