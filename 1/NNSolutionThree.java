import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Created by Daniel on 2016. 09. 28..
 */
public class NNSolutionThree {
	/*
	Feladat:
	3. Valósítsa meg a hibavisszaterjesztés algoritmusát, és egészítse ki a programot NNSolutionThree néven úgy, hogy az a bemenetként kapott neurális hálózat architektúra leírás, neurális hálózat súlyok és bemeneti értékek alapján kiszámolja a neurális hálózat egyes súlyainak és biasainak hatását a kimenetre nézve (parciális deriváltak)!
	*/

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);

		String input = scanner.nextLine();
		System.out.println(input);
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
		for (int i = 0; i < teachingDataNumber; ++i) {
			String[] split1 = scanner.nextLine().split(",");
			List<Double> teachingData = new ArrayList<>();
			for (String s : split1) {
				teachingData.add(Double.parseDouble(s));
			}
			// Első, azaz bemeneti rétegnél az output a bemenet
			neuralNet.getLayer(0).setOutputValues(teachingData);

			// Többi rétegnél már kell szummázni, meg aktivációs függvénnyel számolni
			for (int j = 1; j < neuralNet.getLayerNumber() - 1; ++j) {
				List<Double> jLayerOutputValues = new ArrayList<>();
				for (Neuron neuron : neuralNet.getLayer(j)) {
					jLayerOutputValues.add(Neuron.ReLU(neuron.calculateOutput(teachingData)));
				}
				teachingData = jLayerOutputValues;
			}
		}
		// Delták kiszámítása
		neuralNet.calculateDeltas();
		neuralNet.calculateDerivatives();

		neuralNet.printDerivatives();
	}
}
