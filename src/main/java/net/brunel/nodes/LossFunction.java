package net.brunel.nodes;

public interface LossFunction {

	double computeDerivative(int component, double[] features, double actual, double predicted);

	/**
	 * compute individual error score per component 
	 * 
	 * @param actualLabels
	 * @param predictedLabels
	 */
	double[] computeLossesForInstance(double[] actualLabels, double[] predictedLabels);

	/**
	 * @param the predictions
	 * @param labels the labels
	 * 
	 */
	double computeLoss(double[][] predictions, double[][] labels);

}
