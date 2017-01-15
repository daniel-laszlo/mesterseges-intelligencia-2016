import java.util.*;

/**
 * Created by Daniel on 2016. 09. 11..
 */
public class NeuralNet {
	public static Double epochNumber;
	public static Double learningRate;
	public static Double teachingDataRate;

	private List<Layer> layerList;
	private Set<Double> weights;

	public NeuralNet() {
		layerList = new ArrayList<>();
		weights = new HashSet<>();
	}

	public void addNewLayer(int layerDimension) {
		layerList.add(new Layer(layerDimension));
	}

	public void printWeights() {
		layerList.forEach(Layer::printWeights);
	}

	public void generateRandomNeuronWeights() {
		for (int i = 1; i < layerList.size(); ++i) {
			int previousLayerDimension = layerList.get(i - 1).getDimension();
			layerList.get(i).generateRandomNeuronWeights(previousLayerDimension + 1, weights);
		}
	}

	public int getLayerNumber() {
		return layerList.size();
	}

	public Layer getLayer(int index) {
		return layerList.get(index);
	}

	public void calculateDeltas() {
		layerList.get(layerList.size() - 1).setOutputLayerDeltasToZero();
		for (int i = layerList.size() - 2; i > 0; i--) {
			Layer actualLayer = layerList.get(i);
			Layer nextLayer = layerList.get(i + 1);
			actualLayer.calculateDeltas(nextLayer.getDeltaList(), nextLayer);
		}
	}

	public void calculateDerivatives() {
		for (int i = 1; i < layerList.size(); i++) {
			layerList.get(i).calculateDerivatives(layerList.get(i - 1).getOutputValues(i == 1));
		}
	}

	public void printDerivatives() {
		for (int i = 1; i < layerList.size(); i++) {
			layerList.get(i).printDerivatives();
		}
	}

	public List<Double> forward(List<Double> inputList) {
		layerList.get(0).setOutputValues(inputList);
		for (int i = 1; i < layerList.size(); i++) {
			inputList = layerList.get(i).calculateOutputValues(inputList, i == layerList.size()-1);
		}
		return inputList;
	}

	public void backpropagation(List<Double> error) {
		setLayerDelta(error, layerList.size() - 1);

		for (int i = layerList.size() - 2; i >= 0; i--) {
			// deriváltak
			layerList.get(i + 1).calculateDerivatives(layerList.get(i).getOutputValues(i == 0));
			// delták
			layerList.get(i).calculateDeltas(layerList.get(i + 1).getDeltaList(), layerList.get(i + 1));
		}
	}

	private void setLayerDelta(List<Double> deltas, int index) {
		layerList.get(index).setOutputLayerDeltas(deltas);
	}

	public void modifyWeights() {
		for (int i = 1; i < layerList.size(); i++) {
			layerList.get(i).modifyWeights();
		}
	}

	public List<Double> getLastLayerOutput() {
		return layerList.get(layerList.size() - 1).getOutputValues(true);
	}

	public void loadDataMap(Map dataMap, int startIndex, int endIndex, Scanner scanner) {
		for (int i = startIndex; i < endIndex; ++i) {
			List<Double> inputData = new ArrayList<>();
			List<Double> expectedOutputData = new ArrayList<>();

			String[] inputSplit = scanner.nextLine().split(",");
			for (int j = 0; j < inputSplit.length - layerList.get(layerList.size() - 1).getDimension(); j++) {
				inputData.add(Double.parseDouble(inputSplit[j]));
			}
			for (int j = inputSplit.length - layerList.get(layerList.size() - 1).getDimension(); j < inputSplit.length; j++) {
				expectedOutputData.add(Double.parseDouble(inputSplit[j]));
			}
			dataMap.put(inputData, expectedOutputData);
		}
	}
}
