package net.brunel.nodes;

public class LossFunctionHelper {

	private static class MSELossFunction implements LossFunction {

		MSELossFunction() {
		}

		@Override
		public double computeDerivative(int component, double[] features, double actualLabel, double predictedLabel) {
			return (predictedLabel-actualLabel);
		}

		public double computeSingleLoss(double actual, double predicted) {
			double diff = actual - predicted;
			return diff * diff;
		}

		@Override
		public double[] computeLossesForInstance(double[] actualLabels, double[] predictedLabels) {
			double iterationErrors[] = new double[actualLabels.length];
			for (int j = 0; j < actualLabels.length; j++)
				iterationErrors[j] += computeSingleLoss(actualLabels[j], predictedLabels[j]);
			return iterationErrors;

		}

		public double computeLoss(double[] actualLabels, double[] predictedLabels) {
			double[] individualErrors = computeLossesForInstance(actualLabels, predictedLabels);
			double globalLoss = 0;
			for (double d : individualErrors)
				globalLoss += d;
			return globalLoss;
		}

		@Override
		public double computeLoss(double[][] labels, double[][] predictions) {
			double iterationErrorSum = 0;
			for (int i = 0; i < predictions.length; i++) {
//				double[] iterationErrors = computeIndividualLossDistribution(labels[i], predictions[i]);
//				debug("iterationErrors for instance " + i + " = " + Arrays.toString(iterationErrors));
				iterationErrorSum  += computeLoss(labels[i], predictions[i]);
			}
//			debug("iterationErrorSum = " + iterationErrorSum);
//			return iterationErrorSum / (double)labels.length;
			return iterationErrorSum ;
		}

	}

	private static class CrossEntropyLoss implements LossFunction {

		CrossEntropyLoss() { }

		@Override
		public double computeDerivative(int component, double[] features, double actual, double predicted) {
			double a = features[component] * (predicted-actual);
			return  a;
		}

		
		public double computeSingleLoss(double actual, double predicted) {
			double a1 = actual * Math.log(predicted);
			double a2 = (1-actual) * Math.log(1-predicted);
			
			return a1+a2;
		}

		@Override
		public double[] computeLossesForInstance(double[] actualLabels, double[] predictedLabels) {
			double[] returnValue = new double[actualLabels.length];
			for (int i = 0; i < actualLabels.length; i++) {
				returnValue[i] = computeSingleLoss(actualLabels[i], predictedLabels[i]);
			}
			return returnValue;
		}

		@Override
		public double computeLoss(double[][] labels, double[][] predictions) {
			double loss = 0;
			for(int i = 0; i < predictions.length; i++) {
				double[] distribution = computeLossesForInstance(labels[i], predictions[i]);
				
				for (double d : distribution)
					loss += d;
			}
			
			return (-1/((double)predictions.length)) * loss;
		}


	}

	public static LossFunction MSE_LOSS;
	public static LossFunction CROSS_ENTROPY_LOSS;

	static {
		MSE_LOSS = new MSELossFunction();
		CROSS_ENTROPY_LOSS = new CrossEntropyLoss();
	}

}
