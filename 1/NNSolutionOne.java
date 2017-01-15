import java.util.Scanner;

/**
 * Created by Daniel on 2016. 09. 11..
 */
public class NNSolutionOne {
	/*
	Feladat:
	1. Készítsen egy programot NNSolutionOne néven, mely egy adott architektúrájú neurális hálózat súlyait és bias értékeit inicializálja. A súlyok inicializálása történjen nulla várható értékű, 0.1 szórású normális eloszlásból sorsolt véletlen számokkal. A bias értékek 0-val legyenek inicializálva.
	/*

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		String input = scanner.nextLine();
		System.out.println(input);

		String[] split = input.split(",");

		NeuralNet neuralNet = new NeuralNet();
		for (String str : split) {
			neuralNet.addNewLayer(Integer.parseInt(str));
		}
		neuralNet.generateRandomNeuronWeights();
		neuralNet.printWeights();
	}
}
