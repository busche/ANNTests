package net.brunel.nodes;

import net.brunel.nodes.exceptions.InputException;

public interface Node {

	double computeNodeOutput(double[] acticationsOfPreviousLayer) throws InputException;

	/**
	 * the weight from source node c
	 * 
	 * @param c
	 */
	double getWeightFromInput(int c);

	double w(int k);

	double b();

	void updateW(int dimension, double gradientValue);

	void updateB(double gradientValue);
	
	void prepareUpdate();
	
	void commitUpdate(double learningRate, int datasetsize);
	
	Function getFunction();

	/**
	 * if updates have been prepared, internally use the updated weights but wait until the changes are actually committed. Helps to check different Learning rates.  
	 * 
	 * @param d
	 */
	void configureUpdate(double d, int datasetsize);

	void resetUpdate();

	double computeDerivativeValue(double[] input);

}