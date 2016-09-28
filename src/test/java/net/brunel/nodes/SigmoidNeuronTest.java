package net.brunel.nodes;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import net.brunel.nodes.exceptions.InputDimensionMismatchException;
import net.brunel.nodes.exceptions.InputException;

public class SigmoidNeuronTest {

	@Before
	public void setup() {
	}

	@Test
	public void test() throws InputException {
		Node n1 = new SigmoidNeuron(2, InitializerHelper.newConstantInitializer(0));
		double output = n1.computeOutput(new double[] { 0.1, 0.2 });
		// effectively the sigmoid of 0
		assertEquals(0.5, output, 0.001);
	}

	@Test(expected = InputDimensionMismatchException.class)
	public void mismatchingDimensionality() throws InputException {
		new SigmoidNeuron(0, InitializerHelper.newConstantInitializer(0)).computeOutput(new double[] { 0.1 });

		// fail("Not yet implemented");
	}

}
