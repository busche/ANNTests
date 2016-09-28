package net.brunel.nodes;

public class InitializerHelper {

	public static Initializer newConstantInitializer(int i) {
		return (min, max) -> {
			return i;
		};
	}

	public static Initializer newCircularInitializer(double[] ds) {
		return new Initializer() {
			int idx=0;
			
			@Override
			public double nextDouble(int i, int j) {
				double returnValue = ds[idx++];
				if (idx >= ds.length)
					idx=0;
				return returnValue;
			}
		};
	}

}
