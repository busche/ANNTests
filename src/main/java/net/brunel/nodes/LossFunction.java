package net.brunel.nodes;

public interface LossFunction {

	double computeDerivative(double actual, double predicted);

	double computeValue(double actual, double predicted);

}
