package net.brunel.nodes.exceptions;

public class InputDimensionMismatchException extends InputException {

	private int expectedDimensionality;
	private int givenDimensionality;

	public InputDimensionMismatchException(int expected, int given) {
		super("Given dimensionality (" + given + ") does not match the expected one (" + expected + ")");
		this.expectedDimensionality = expected;
		this.givenDimensionality = given;
	}

}
