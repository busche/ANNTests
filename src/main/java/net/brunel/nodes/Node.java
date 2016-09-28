package net.brunel.nodes;

import net.brunel.nodes.exceptions.InputException;

public interface Node {

	double computeOutput(double[] input) throws InputException;

	/**
	 * the weight from source node c
	 * 
	 * @param c
	 */
	double getWeightFromInput(int c);

	double computeDerivative(double[] instanceData, int dimension);

	double w(int k);

	double b();

	void updateW(double learningRate, int dimension, double gradientValue);

	void updateB(double learningRate, double gradientValue);

}