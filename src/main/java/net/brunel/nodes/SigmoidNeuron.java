package net.brunel.nodes;

import net.brunel.nodes.exceptions.InputDimensionMismatchException;
import net.brunel.nodes.exceptions.InputException;

public class SigmoidNeuron implements Node {

	private static final int INITIALIZATION_MINIMUM = -1;
	private static final int INITIALIZATION_MAXIMUM = 1;

	private double[] weights;
	private double bias;

	public SigmoidNeuron(int numberOfInputs, Initializer initializer) {
		super();
		weights = new double[numberOfInputs];
		for (int i = 0; i < numberOfInputs; i++)
			weights[i] = initializer.nextDouble(INITIALIZATION_MINIMUM, INITIALIZATION_MAXIMUM);
		bias = initializer.nextDouble(INITIALIZATION_MINIMUM, INITIALIZATION_MAXIMUM);
	}

	/* (non-Javadoc)
	 * @see net.brunel.nodes.Node#compute(double[])
	 */
	@Override
	public double computeOutput(double[] input) throws InputException {
		if (input.length != weights.length)
			throw new InputDimensionMismatchException(weights.length, input.length);

		double dotProduct = MyMath.dotProduct(weights, input);

		return MyMath.sigmoid(dotProduct + bias);
	}

	@Override
	public double getWeightFromInput(int c) {
		return weights[c];
	}

	@Override
	public double computeDerivative(double[] input, int dimension) {
		double dotProduct = MyMath.dotProduct(weights, input);
		double z = dotProduct + bias;

		return MyMath.sigmoid(z)*(1-MyMath.sigmoid(z));
	}

	@Override
	public double w(int k) {
		return weights[k];
	}

	@Override
	public double b() {
		return bias;
	}

	@Override
	public void updateW(double learningRate, int dimension, double gradientValue) {
		weights[dimension] -= learningRate*gradientValue;
	}

	@Override
	public void updateB(double learningRate, double gradientValue) {
		bias -= learningRate*gradientValue;
	}

}
