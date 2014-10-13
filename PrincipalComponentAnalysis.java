import java.util.*;
import java.io.*;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.linear.SingularValueDecomposition;

class PrincipalComponentAnalysis {
	private RealMatrix examples, covariance;
	public static void main(String[] args) throws IOException {
		PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();
		pca.examples = pca.readExamples("examples.txt");
		pca.examples = pca.normalize(pca.examples);
		pca.covariance = pca.calculateCovariance(pca.examples);
		
		SingularValueDecomposition svd = new SingularValueDecomposition(pca.covariance);
		RealMatrix u = svd.getU(), s = svd.getS(), v = svd.getV();
		int k = pca.calculateK(pca.examples,s);
		RealMatrix subU = PrincipalComponentAnalysis.reduceMatrix(u,k);
		RealMatrix reducedExamples = pca.calculatePCA(subU,pca.examples);
		System.out.println("The sum of squared errors is: " + pca.calculateError(pca.examples,reducedExamples,subU));
		pca.printExamples(reducedExamples,"reducedExamples.txt");
		
	}
	
	// Returns the first k columns of mat.
	public static RealMatrix reduceMatrix(RealMatrix mat, int k) {
		double[][] buffer = new double[mat.getRowDimension()][k];
		mat.copySubMatrix(0,mat.getRowDimension()-1,0,k-1,buffer);
		RealMatrix subU = new BlockRealMatrix(buffer);	
		return subU;
	}
	
	// Calculates the error per example's feature as the difference squared.
	private double calculateError(RealMatrix examples, RealMatrix z, RealMatrix u) {
		RealMatrix approximation = z.multiply(u.transpose());
		double error = 0;
		for( int i = 0; i < examples.getRowDimension(); i++ ) {
			for( int j = 0; j < examples.getColumnDimension(); j++ ) {
				error += Math.pow(examples.getEntry(i,j)-approximation.getEntry(i,j),2);
			}
		}
		return error;
	}
	
	// Calculates the final step in the algorithm.
	private RealMatrix calculatePCA(RealMatrix u, RealMatrix examples) {
		RealMatrix z = examples.multiply(u);
		return z;
	}
	
	// Calculates the minimum optimal value for k.
	private int calculateK(RealMatrix examples, RealMatrix s) {
		int numExamples = examples.getRowDimension();
		boolean foundK = false;
		int k = 1;
		double buffer = 0, numerator, denominator;
		while(!foundK) {
			for( int i = 0; i < k; i++ )
				buffer = buffer + s.getEntry(i,i);
				
			numerator = buffer;
				
			for( int j = k; j < numExamples; j++ ) 
				buffer = buffer + s.getEntry(j,j);
				
			denominator = buffer;
			if(numerator/denominator >= 0.99)
				foundK = true;
			else 
				k++;
		}
		return k;
	}
	
	// Reads in the examples.
	private RealMatrix readExamples(String fileName) throws IOException {
		List<List<Double>> list = new ArrayList<List<Double>>();
		File f = new File(fileName);
		BufferedReader br = new BufferedReader(new FileReader(f));
		String line;
		while((line = br.readLine()) != null) {
			List<Double> convertedStrings = new ArrayList<Double>();
			List<String> strings = Arrays.asList(line.split("\\s+"));
			for( String s : strings )
				convertedStrings.add(Double.parseDouble(s));
				
			list.add(convertedStrings);
		}	
		int numExamples = list.size(), numFeatures = list.get(0).size();
		BlockRealMatrix examples = new BlockRealMatrix(numExamples,numFeatures);
		for( int i = 0; i < numExamples; i++ ) {
			Double[] example = new Double[numFeatures];
			list.get(i).toArray(example);
			examples.setRowVector(i,new ArrayRealVector(example));
		}
		return examples;
	}
	
	// Prints out the given RealMatrix to file.
	private void printExamples(RealMatrix examples, String fileName) throws IOException {
		PrintWriter pw = new PrintWriter(fileName);
		StringBuilder sb = new StringBuilder("");
		for( int i = 0; i < examples.getRowDimension(); i++ ) {
			for( int j = 0; j < examples.getColumnDimension(); j++ ) { 
				sb.append(examples.getEntry(i,j) + " ");
			}
			pw.println(sb);
			sb = new StringBuilder("");
		}
		pw.flush();
		pw.close();
	}
	
	// Normalizes the examples.
	private RealMatrix normalize(RealMatrix examples) {
		ArrayRealVector mean = this.calculateStatistic(examples, new Mean()), stddev = this.calculateStatistic(examples, new StandardDeviation());
		int numExamples = examples.getRowDimension(), numFeatures = examples.getColumnDimension();
		for( int i = 0; i < numExamples; i++ ) {
			RealVector example = examples.getRowVector(i);
			example = example.subtract(mean).ebeDivide(stddev);
			examples.setRowVector(i,example);
		}
		return examples;
	}
	
	// Calculates the given statistic per feature.
	private ArrayRealVector calculateStatistic(RealMatrix examples, AbstractStorelessUnivariateStatistic statistic) throws IllegalArgumentException {
		int rows = examples.getRowDimension(), cols = examples.getColumnDimension();
		double[] means = new double[cols];
		for( int i = 0; i < cols; i++ ) {
			double[] features = examples.getColumnVector(i).toArray();
			means[i] = statistic.evaluate(features,0,features.length);
		}
		return new ArrayRealVector(means);
	}	
	
	// Calculates the covariance matrix given the examples.
	private RealMatrix calculateCovariance(RealMatrix examples) {
		return new Covariance(examples).getCovarianceMatrix();
	}
}