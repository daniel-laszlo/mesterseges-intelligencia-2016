import java.util.*;

/**
 * Created by Daniel on 2016. 10. 13..
 */
public class NNSolutionFour {
	
	/*
	Feladat:
	4. Valósítsa meg a neurális hálózat tanítását. Egészítse a programot NNSolutionFour néven úgy, hogy az képes legyen a bemenetként kapott neurális hálózat architektúra leírás, tanítási paraméterek, neurális hálózat súlyok, valamint egy taníttóminta-készlet alapján a hálózat tanítására és validációjára!
	*/

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		NeuralNet neuralNet = new NeuralNet();

		// Tanítási paraméterek
		String input = scanner.nextLine();
		String[] teachingParameters = input.split(",");
		NeuralNet.epochNumber = Double.parseDouble(teachingParameters[0]);
		NeuralNet.learningRate = Double.parseDouble(teachingParameters[1]);
		NeuralNet.teachingDataRate = Double.parseDouble(teachingParameters[2]);

		// A háló architektúrájának kialakítása
		String architecture = scanner.nextLine();
		String[] split = architecture.split(",");
		for (int i = 0; i < split.length; ++i) {
			int layerDimension = Integer.parseInt(split[i]);
			neuralNet.addNewLayer(layerDimension);
		}

		// Súlyok feltöltése
		for (int i = 1; i < neuralNet.getLayerNumber(); ++i) {
			for (Neuron neuron : neuralNet.getLayer(i)) {
				List<Double> inputWeightList = new ArrayList<>();
				for (String s : scanner.nextLine().split(",")) {
					inputWeightList.add(Double.parseDouble(s));
				}
				neuron.setWeightList(inputWeightList);
			}
		}

		// Tanító és validációs adatok beolvasása
		int inputDataNumber = Integer.parseInt(scanner.nextLine());
		int teachingDataMaxIndex = (int) Math.floor(inputDataNumber * NeuralNet.teachingDataRate);

		Map<List<Double>, List<Double>> teachingDataMap = new LinkedHashMap<>();
		Map<List<Double>, List<Double>> validationDataMap = new LinkedHashMap<>();
		neuralNet.loadDataMap(teachingDataMap, 0, teachingDataMaxIndex, scanner);
		neuralNet.loadDataMap(validationDataMap, teachingDataMaxIndex, inputDataNumber, scanner);

		// Epochok
		for (int i = 0; i < NeuralNet.epochNumber; i++) {

			// Tanító adatok
			for (Map.Entry<List<Double>, List<Double>> aTeachingData : teachingDataMap.entrySet()) {
				List<Double> errorWithOneTeachingData = difference(aTeachingData.getValue(), neuralNet.forward(aTeachingData.getKey()));
				neuralNet.backpropagation(errorWithOneTeachingData);
				neuralNet.modifyWeights();
			}

			List<Double> validationErrors = new ArrayList<>();
			// Validációs adatok
			for (Map.Entry<List<Double>, List<Double>> aValidationData : validationDataMap.entrySet()) {
				neuralNet.forward(aValidationData.getKey());
				validationErrors.addAll(difference(aValidationData.getValue(), neuralNet.getLastLayerOutput()));
			}

			// Validációs hibák kiírása
			Double error = 0.0;
			for (Double validationError : validationErrors) {
				error += validationError * validationError;
			}
			System.out.println(error / validationErrors.size());
		}
		System.out.println(architecture);
		neuralNet.printWeights();
	}

	public static List<Double> difference(List<Double> one, List<Double> two) {
		List<Double> ret = new ArrayList<>();
		for (int i = 0; i < one.size(); i++) {
			ret.add(one.get(i) - two.get(i));
		}
		return ret;
	}
}
