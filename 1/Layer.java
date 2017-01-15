import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by Daniel on 2016. 09. 11..
 */
public class Layer implements Iterable<Neuron> {
	public static final Integer MEAN = 0;
	public static final Double STANDARD_DEVIATION = 0.1;

	private int dimension; // hány db neuront tartalmaz a réteg
	private List<Neuron> neuronList;

	public Layer(int dimensions) {
		this.dimension = dimensions;
		neuronList = new ArrayList<>();
		for (int i = 0; i < dimensions; ++i) {
			neuronList.add(new Neuron());
		}
	}

	public void generateRandomNeuronWeights(int numberOfWeightsForEachNeuron, Set<Double> ignoreWeightSet) {
		for (int i = 0; i < dimension; ++i) {
			List<Double> uniqueWeightList = new ArrayList<>();
			boolean isListUnique;
			do {
				uniqueWeightList = generateWeightList(numberOfWeightsForEachNeuron);
				isListUnique = true;
				for (Double aDouble : uniqueWeightList) {
					if (ignoreWeightSet.contains(aDouble)) {
						isListUnique = false;
					}
				}
			} while (!isListUnique);

			ignoreWeightSet.addAll(uniqueWeightList);
			ignoreWeightSet.remove(0.0);
			Neuron neuron = new Neuron(uniqueWeightList);
			neuronList.add(neuron);
		}
	}

	private List<Double> generateWeightList(int numberOfWeights) {
		List<Double> ret = new ArrayList<>();
		Random r = new Random();

		for (int i = 0; i < numberOfWeights - 1; ++i) {
			boolean isGeneratedNumberUnique = false;
			double generatedNumber = 8000;
			while (!isGeneratedNumberUnique) {
				generatedNumber = round(r.nextGaussian() * STANDARD_DEVIATION + MEAN, 8);
				if (!ret.contains(generatedNumber)) {
					isGeneratedNumberUnique = true;
				}
			}
			ret.add(generatedNumber);
		}
		ret.add(0.0);

		return ret;
	}

	private static double round(double value, int precision) {
		int scale = (int) Math.pow(10, precision);
		return (double) Math.round(value * scale) / scale;
	}

	public int getDimension() {
		return dimension;
	}

	public Neuron getNeuron(int index) {
		return neuronList.get(index);
	}

	public void printWeights() {
		neuronList.forEach(Neuron::printWeights);
	}

	@Override
	public Iterator<Neuron> iterator() {
		return neuronList.iterator();
	}

	public void calculateDeltas(List<Double> nextLayerDeltaList, Layer nextLayer) {
		for (int i = 0; i < neuronList.size(); i++) {
			neuronList.get(i).calculateDelta(nextLayerDeltaList, nextLayer.getWeightList(i));
		}
	}

	public void setOutputLayerDeltasToZero() {
		for (Neuron neuron : neuronList) {
			neuron.setDelta(1.0);
		}
	}

	public void setOutputLayerDeltas(List<Double> deltas) {
		for (int i = 0; i < neuronList.size(); i++) {
			neuronList.get(i).setDelta(deltas.get(i));
		}
	}


	public List<Double> getDeltaList() {
		List<Double> ret = neuronList.stream().map(Neuron::getDelta).collect(Collectors.toList());
		return ret;
	}

	public List<Double> getOutputValues(boolean isSpecialLayer) {
		List<Double> ret = new ArrayList<>();
		if (isSpecialLayer) {
			ret.addAll(neuronList.stream().map(Neuron::getSum).collect(Collectors.toList()));
		} else {
			ret.addAll(neuronList.stream().map(neuron -> Neuron.ReLU(neuron.getSum())).collect(Collectors.toList()));
		}
		return ret;
	}

	public void calculateDerivatives(List<Double> previousLayerOutputList) {
		for (Neuron neuron : neuronList) {
			neuron.calculateDerivatives(previousLayerOutputList);
		}
	}

	public void printDerivatives() {
		neuronList.forEach(Neuron::printDerivatives);
	}

	public List<Double> getWeightList(int index) {
		return neuronList.stream().map(neuron -> neuron.getWeight(index)).collect(Collectors.toList());
	}

	public void setOutputValues(List<Double> outputValues) {
		for (int i = 0; i < outputValues.size(); ++i) {
			neuronList.get(i).setSum(outputValues.get(i));
		}
	}

	public void modifyWeights() {
		for (Neuron neuron : neuronList) {
			neuron.modifyWeights();
		}
	}

	public List<Double> calculateOutputValues(List<Double> input, boolean isSpecialLayer) {
		List<Double> outputValues = new ArrayList<>();
		for (Neuron neuron : neuronList) {
			if (isSpecialLayer) {
				outputValues.add(neuron.calculateOutput(input));
			} else {
				outputValues.add(Neuron.ReLU(neuron.calculateOutput(input)));
			}
		}
		return outputValues;
	}
}
