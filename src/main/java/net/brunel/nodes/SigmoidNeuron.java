package net.brunel.nodes;

import java.util.Arrays;

import net.brunel.nodes.exceptions.InputDimensionMismatchException;
import net.brunel.nodes.exceptions.InputException;

public class SigmoidNeuron implements Node, Function {

	private static final int INITIALIZATION_MINIMUM = -1;
	private static final int INITIALIZATION_MAXIMUM = 1;

	private double[] weights;
	private double bias;
	double[] updateWeights;
	double updateBias;
	private double configuredUpdateLearningRate;
	private boolean useConfiguredUpdateLearningRate;

	public SigmoidNeuron(int numberOfInputs, Initializer initializer) {
		super();
		weights = new double[numberOfInputs];
		updateWeights = new double[numberOfInputs];
		
		for (int i = 0; i < numberOfInputs; i++)
			weights[i] = initializer.nextDouble(INITIALIZATION_MINIMUM, INITIALIZATION_MAXIMUM);
		bias = initializer.nextDouble(INITIALIZATION_MINIMUM, INITIALIZATION_MAXIMUM);
		useConfiguredUpdateLearningRate=false;
	}

	/* (non-Javadoc)
	 * @see net.brunel.nodes.Node#compute(double[])
	 */
	@Override
	public double computeNodeOutput(double[] input) throws InputException {
		if (input.length != weights.length)
			throw new InputDimensionMismatchException(weights.length, input.length);

		double localBias = bias;
		double[] localWeightVector = weights;
		if (useConfiguredUpdateLearningRate) {
			localWeightVector = new double[weights.length];
			System.arraycopy(weights, 0, localWeightVector, 0, weights.length);
			for (int i = 0; i < weights.length; i++)
				localWeightVector[i] = w(i);
			localBias=b();
		}
		double dotProduct = MyMath.dotProduct(localWeightVector, input);

		return computeAt(dotProduct + localBias);
	}
	
	@Override
	public double computeDerivativeValue(double[] input) {
		double localBias = bias;
		double[] localWeightVector = weights;
		if (useConfiguredUpdateLearningRate) {
			localWeightVector = new double[weights.length];
			System.arraycopy(weights, 0, localWeightVector, 0, weights.length);
			for (int i = 0; i < weights.length; i++)
				localWeightVector[i] = w(i);
			localBias=b();
		}
		double dotProduct = MyMath.dotProduct(localWeightVector, input);

		return computeDerivativeValue(dotProduct + localBias);
	}


	@Override
	public double getWeightFromInput(int c) {
		return weights[c];
	}

	@Override
	public double w(int k) {
		double returnValue =  weights[k];
		if (useConfiguredUpdateLearningRate) {
			returnValue += configuredUpdateLearningRate*updateWeights[k];
		}
		return returnValue;
	}

	@Override
	public double b() {
		double returnValue =  bias;
		if (useConfiguredUpdateLearningRate) {
			returnValue += configuredUpdateLearningRate*updateBias;
		}
		return returnValue;
	}

	
	@Override
	public void updateW(int dimension, double gradientValue) {
		updateWeights[dimension] += gradientValue;
//		System.out.println("update request for w(" + dimension + ") = " + gradientValue);
	}

	@Override
	public void updateB(double gradientValue) {
		updateBias +=gradientValue;
	}

	@Override
	public Function getFunction() {
		return this;
	}

	@Override
	public double computeDerivativeValue(double v) {
		return MyMath.sigmoid(v)*(1-MyMath.sigmoid(v));
	}

	@Override
	public double computeAt(double v) {
		return MyMath.sigmoid(v);
	}

	@Override
	public void prepareUpdate() {
		
	}

	@Override
	public void commitUpdate(double learningRate) {
		for (int i = 0; i < weights.length; i++) {
//			System.out.print("weights[i] -= " + learningRate + "*" + updateWeights[i] + " ==> " + weights[i] + " -=  " + (learningRate*updateWeights[i]));
			weights[i] -= learningRate*updateWeights[i];
//			System.out.println(" ==> weights[" + i + "] = " + weights[i]);
		}
		bias -= learningRate * updateBias;
		// reset values
		Arrays.setAll(updateWeights, (a)->{return 0;});
		updateBias=0;
		this.useConfiguredUpdateLearningRate=false;
	}

	@Override
	public void configureUpdate(double d) {
		this.configuredUpdateLearningRate = d;
		this.useConfiguredUpdateLearningRate = true;
	}

	@Override
	public void resetUpdate() {
		useConfiguredUpdateLearningRate=false;
		Arrays.setAll(updateWeights, (a)->{return 0;});
		updateBias=0;
	}


}
