import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Daniel on 2016. 11. 01..
 */
public class Bayes {

	private Double param_Beta;
	private Integer param_I;
	private Integer param_J;
	private Integer param_L;

	private Double alfa_u;
	private Double alfa_v;
	private RealMatrix matrix_H;
	private RealMatrix matrix_U;
	private RealMatrix matrix_V;

	private List<RealMatrix> avg_U = new ArrayList<>();
	private List<RealMatrix> avg_V = new ArrayList<>();

	public Bayes(double b, double[][] h, int i, int j, int l) {
		param_I = i;
		param_J = j;
		param_L = l;
		param_Beta = b;
		matrix_H = new BlockRealMatrix(h);
		alfa_u = 0.0001;
		alfa_v = 0.0001;
		MultivariateNormalDistribution normalDistribution = new MultivariateNormalDistribution(new double[l], MatrixUtils.createRealIdentityMatrix(l).scalarMultiply(1 / alfa_u).getData());
		matrix_U = new BlockRealMatrix(normalDistribution.sample(i));
		matrix_V = new BlockRealMatrix(normalDistribution.sample(j));
	}

	public void doEpoch() {
		for (int i = 0; i < 5; i++) {
			recalculateMatrixU();
			recalculateMatrixV();
		}
		for (int i = 0; i < 20; i++) {
			recalculateMatrixU();
			recalculateMatrixV();
			avg_U.add(matrix_U);
			avg_V.add(matrix_V);
		}
	}

	public void recalculateMatrixU() {
		RealMatrix lambdaInverse = MatrixUtils.inverse(getLambdaForMatrixU());
		for (int i = 0; i < param_I; i++) {
			MultivariateNormalDistribution nD = new MultivariateNormalDistribution(getPsiForMatrixU(i).getColumn(0), lambdaInverse.getData());
			matrix_U.setRowVector(i, new ArrayRealVector(nD.sample()));
		}
	}

	public RealMatrix getLambdaForMatrixU() {
		RealMatrix lambda = new BlockRealMatrix(param_L, param_L);
		for (int j = 0; j < param_J; j++) {
			RealMatrix lambdaIncr = matrix_V.getRowMatrix(j).transpose().multiply(matrix_V.getRowMatrix(j));
			lambda = lambda.add(lambdaIncr);
		}
		lambda = lambda.scalarMultiply(param_Beta);
		lambda = lambda.add(MatrixUtils.createRealIdentityMatrix(param_L).scalarMultiply(alfa_u));
		return lambda;
	}

	public RealMatrix getPsiForMatrixU(int i) {
		RealMatrix psi = new BlockRealMatrix(1, param_L);
		for (int j = 0; j < param_J; j++) {
			psi = psi.add(matrix_V.getRowMatrix(j).scalarMultiply(matrix_H.getEntry(i, j)));
		}
		return MatrixUtils.inverse(getLambdaForMatrixU()).scalarMultiply(param_Beta).multiply(psi.transpose());
	}

	public void recalculateMatrixV() {
		RealMatrix lambdaInverse = MatrixUtils.inverse(getLambdaForMatrixV());
		for (int i = 0; i < param_J; i++) {
			MultivariateNormalDistribution nD = new MultivariateNormalDistribution(getPsiForMatrixV(i).getColumn(0), lambdaInverse.getData());
			matrix_V.setRowVector(i, new ArrayRealVector(nD.sample()));
		}
	}

	public RealMatrix getLambdaForMatrixV() {
		RealMatrix lambda = new BlockRealMatrix(param_L, param_L);
		for (int i = 0; i < param_I; i++) {
			lambda = lambda.add(matrix_U.getRowMatrix(i).transpose().multiply(matrix_U.getRowMatrix(i)));
		}
		lambda = lambda.scalarMultiply(param_Beta);
		lambda = lambda.add(MatrixUtils.createRealIdentityMatrix(param_L).scalarMultiply(alfa_v));
		return lambda;
	}

	public RealMatrix getPsiForMatrixV(int j) {
		RealMatrix psi = new BlockRealMatrix(1, param_L);
		for (int i = 0; i < param_I; i++) {
			psi = psi.add(matrix_U.getRowMatrix(i).scalarMultiply(matrix_H.getEntry(i, j)));
		}
		return MatrixUtils.inverse(getLambdaForMatrixV()).scalarMultiply(param_Beta).multiply(psi.transpose());
	}


	public void printOutput() {
		RealMatrix outputMatrix_U = new BlockRealMatrix(matrix_U.getRowDimension(), matrix_U.getColumnDimension());
		RealMatrix outputMatrix_V = new BlockRealMatrix(matrix_V.getRowDimension(), matrix_V.getColumnDimension());
		for (int i = 0; i < avg_U.size(); i++) {
			outputMatrix_U = outputMatrix_U.add(avg_U.get(i));
			outputMatrix_V = outputMatrix_V.add(avg_V.get(i));
		}
		outputMatrix_U = outputMatrix_U.scalarMultiply(1.0 / avg_U.size());
		outputMatrix_V = outputMatrix_V.scalarMultiply(1.0 / avg_V.size());

		RealMatrix matrix_U_T = outputMatrix_U;
		for (int i = 0; i < matrix_U_T.getRowDimension(); i++) {
			for (int j = 0; j < matrix_U_T.getColumnDimension() - 1; j++) {
				System.out.print(matrix_U_T.getEntry(i, j) + ",");
			}
			System.out.println(matrix_U_T.getEntry(i, matrix_U_T.getColumnDimension() - 1));
		}

		System.out.println();

		RealMatrix matrix_V_T = outputMatrix_V;
		for (int i = 0; i < matrix_V_T.getRowDimension(); i++) {
			for (int j = 0; j < matrix_V_T.getColumnDimension() - 1; j++) {
				System.out.print(matrix_V_T.getEntry(i, j) + ",");
			}
			System.out.println(matrix_V_T.getEntry(i, matrix_V_T.getColumnDimension() - 1));
		}
	}
}
