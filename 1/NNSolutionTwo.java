import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Created by Daniel on 2016. 09. 15..
 */
public class NNSolutionTwo {
	/*
	Feladat:
	2. Készítsen egy programot NNSolutionTwo néven, amely a bemenetként kapott neurális hálózat (MLP) architektúra leírás, neurális hálózat súlyok és bemeneti értékek alapján kiszámolja a neurális hálózat kimenetét!
	*/

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);

		String input = scanner.nextLine();
		String[] split = input.split(",");
		NeuralNet neuralNet = new NeuralNet();

		// A háló architektúrájának kialakítása
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

		// Tanító adatok feldolgozása
		int teachingDataNumber = Integer.parseInt(scanner.nextLine());
		System.out.println(teachingDataNumber);
		for (int i = 0; i < teachingDataNumber; ++i) {
			String[] split1 = scanner.nextLine().split(",");
			List<Double> teachingData = new ArrayList<>();
			for (String s : split1) {
				teachingData.add(Double.parseDouble(s));
			}
			for (int j = 1; j < neuralNet.getLayerNumber() - 1; ++j) {
				List<Double> jLayerOutputValues = new ArrayList<>();
				for (Neuron neuron : neuralNet.getLayer(j)) {
					jLayerOutputValues.add(Neuron.ReLU(neuron.calculateOutput(teachingData)));
				}
				teachingData = jLayerOutputValues;
			}

			// Számítások és kiíratás stdOutra
			Layer lastLayer = neuralNet.getLayer(neuralNet.getLayerNumber() - 1);
			for (int j = 0; j < lastLayer.getDimension() - 1; ++j) {
				System.out.print(lastLayer.getNeuron(j).calculateOutput(teachingData) + ",");
			}
			System.out.println(lastLayer.getNeuron(lastLayer.getDimension() - 1).calculateOutput(teachingData));
		}
	}
}
