import java.util.ArrayList;
import java.util.List;

/**
 * Created by Daniel on 2016. 09. 11..
 */
public class Neuron {

	private List<Double> weightList;
	private List<Double> derivativeList;
	private Double delta;
	private Double sum;

	public Neuron() {
		derivativeList = new ArrayList<>();
	}

	public Neuron(List<Double> weights) {
		this.weightList = weights;
	}

	public void setWeightList(List<Double> weightList) {
		this.weightList = weightList;
	}

	public void printWeights() {
		if (weightList != null) {
			for (int i = 0; i < weightList.size() - 1; ++i) {
				System.out.print(weightList.get(i) + ",");
			}
			System.out.println(weightList.get(weightList.size() - 1));
		}
	}

	private Double getBiasValue() {
		return weightList.get(weightList.size() - 1);
	}

	public double calculateOutput(List<Double> inputValueList) {
		double ret = 0.0;
		for (int i = 0; i < inputValueList.size(); ++i) {
			ret += inputValueList.get(i) * weightList.get(i);
		}
		ret += getBiasValue();
		sum = ret;
		return ret;
	}

	public static Double ReLU(Double sum) {
		return Math.max(sum, 0);
	}

	public static Double dReLU(Double value) {
		return (value > 0) ? 1.0 : 0.0;
	}

	public void setDelta(Double delta) {
		this.delta = delta;
	}

	public Double getDelta() {
		return delta;
	}

	public Double getSum() {
		return sum;
	}

	public void calculateDelta(List<Double> nextLayerDeltaList, List<Double> nextLayerWeightList) {
		Double delta = 0.0;

		for (int i = 0; i < nextLayerDeltaList.size(); i++) {
			delta += nextLayerDeltaList.get(i) * nextLayerWeightList.get(i);
		}
		this.delta = delta * dReLU(sum);
	}

	public void calculateDerivatives(List<Double> previousLayerOutputList) {
		derivativeList.clear();
		for (int i = 0; i < previousLayerOutputList.size(); i++) {
			derivativeList.add(previousLayerOutputList.get(i) * delta);
		}
		derivativeList.add(delta);
	}

	public void printDerivatives() {
		for (int i = 0; i < derivativeList.size() - 1; i++) {
			System.out.print(derivativeList.get(i) + ",");
		}
		System.out.println(derivativeList.get(derivativeList.size() - 1));
	}

	public Double getWeight(int index) {
		return weightList.get(index);
	}

	public void setSum(Double sum) {
		this.sum = sum;
	}

	public void modifyWeights() {
		List<Double> newWeightList = new ArrayList<>();
		for (int i = 0; i < weightList.size(); i++) {
			newWeightList.add(weightList.get(i) + derivativeList.get(i) * 2 * NeuralNet.learningRate);
		}
		weightList = newWeightList;
	}
}
