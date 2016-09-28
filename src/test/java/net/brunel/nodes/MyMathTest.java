package net.brunel.nodes;

import static org.junit.Assert.*;

import org.junit.Test;

public class MyMathTest {

	@Test
	public void testDotProduct() {
		double value = MyMath.dotProduct(new double[] { 1, 2, 3 }, new double[] { 1, 2, 3 });
		assertEquals(14, value, 0.001);
		// fail("Not yet implemented");
	}

	@Test
	public void testSigmoid() {
		assertEquals(0.5, MyMath.sigmoid(0), 0.01);
		// fail("Not yet implemented");
	}

	@Test
	public void testPositiveSigmoid() {
		assertEquals(1, MyMath.sigmoid(1000000000), 0.01);
	}

	@Test
	public void testNegativeSigmoid() {
		assertEquals(0, MyMath.sigmoid(-1000000000), 0.01);
	}

	@Test
	public void testConstructor() {
		MyMath mm = new MyMath();
		assertTrue(mm != null);
	}
}
