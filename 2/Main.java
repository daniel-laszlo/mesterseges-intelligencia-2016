import java.util.Scanner;

/**
 * Created by Daniel on 2016. 11. 01..
 */
public class Main {

	public static void main(String[] args) {

		// IN
		Scanner scanner = new Scanner(System.in);
		String[] params = scanner.nextLine().split(",");
		int I = Integer.parseInt(params[0]);
		int J = Integer.parseInt(params[1]);
		int L = Integer.parseInt(params[2]);
		double beta = Double.parseDouble(params[3]);
		double[][] matrix = new double[I][J];

		for (int i = 0; i < I; i++) {
			String[] row = scanner.nextLine().split(",");
			for (int j = 0; j < J; j++) {
				matrix[i][j] = Double.parseDouble(row[j]);
			}
		}
		Bayes bayes = new Bayes(beta, matrix, I, J, L);

		// CALC
		bayes.doEpoch();

		// OUT
		bayes.printOutput();
	}
}
