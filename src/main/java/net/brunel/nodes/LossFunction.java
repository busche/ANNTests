package net.brunel.nodes;

public interface LossFunction {

	double computeLoss(double actualLabel, double predictedLabel);

}
