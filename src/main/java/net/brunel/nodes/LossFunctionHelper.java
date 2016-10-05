package net.brunel.nodes;

public class LossFunctionHelper {

	private static class MSELossFunction implements LossFunction {

		MSELossFunction() {
			// TODO Auto-generated constructor stub
		}

		@Override
		public double computeDerivative(double actual, double predicted) {
			return (actual-predicted);
		}

		@Override
		public double computeValue(double actual, double predicted) {
			double diff = actual - predicted; 
			return diff*diff;
		}
		
	}
	
	public static LossFunction MSE_LOSS;
	
	static {
		MSE_LOSS = new MSELossFunction();
	}
	
}
