package net.brunel.nodes;

public class MyMath {

	public MyMath() {
		
	}
	public static double dotProduct(double[] a, double[] b) {
		double ret = 0;
		for (int i = 0; i < a.length; i++)
			ret += a[i] * b[i];
		return ret;
	}

	public static double sigmoid(double d) {
		return (1/(1 + Math.exp(-d)));
	}

}
