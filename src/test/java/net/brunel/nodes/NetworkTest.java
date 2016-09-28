package net.brunel.nodes;

import static org.junit.Assert.*;

import java.io.PrintStream;
import java.util.Arrays;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import net.brunel.nodes.exceptions.InputDimensionMismatchException;
import net.brunel.nodes.exceptions.InputException;
import net.brunel.nodes.exceptions.NetworkLayerException;

public class NetworkTest {

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void testConstructor() {
		Network n = new Network(2, 1); // 2 input dimensions, one layer, 1
											// output dimension

		assertEquals(2, n.getNumberOfInputs());
		assertEquals(2, n.getNumberOfLayers());
	}
	
	@Test(expected=InputDimensionMismatchException.class)
	public void testMismatchingInputDimensions() throws InputException  {
		Network n = new Network(2, 1); // 2 input dimensions, one layer, 1
											// output dimension
		n.feedForward(new double[] {});
	}
	
	@Test()
	public void testDebugOn() throws InputException  {
		Network n = new Network(2, 1); // 2 input dimensions, one layer, 1
											// output dimension
		n.setDebugOn(true);
		assertEquals(true, n.isDebugOn());
		n.setDebugOn(false);
		assertEquals(false, n.isDebugOn());
	}

	@Test
	public void testFeedin1Layer() throws InputException {
		Network n = new Network(2, 1); // 2 input dimensions, one layer
		
		n.configureLayer(1, new SigmoidNeuron[] { 
				new SigmoidNeuron(2, InitializerHelper.newConstantInitializer(1)),
				new SigmoidNeuron(2, InitializerHelper.newConstantInitializer(1)), });

		for (double[] currentTuple : new double[][] { 
				new double[] { -3, 2 }, 
				new double[] { -2, 1 },
				new double[] { -1, 0 } }) {
			double[] label = n.feedForward(currentTuple);

			assertArrayEquals(
					new double[] { 0.5, 0.5 }, 
					label, 0.01);
		}
	}

	@Test
	public void testFeedin2Layers() throws InputException {
		Network n = new Network(2, 2); // 2 input dimensions, one layer

		n.configureLayer(1, new SigmoidNeuron[] { 
				new SigmoidNeuron(2, InitializerHelper.newConstantInitializer(1)),
				new SigmoidNeuron(2, InitializerHelper.newConstantInitializer(1)), });

		n.configureLayer(2, new SigmoidNeuron[] { 
				new SigmoidNeuron(2, InitializerHelper.newConstantInitializer(1)),
				});

		double[] label = n.feedForward(new double[]{ -2, 1 });
//		assertArrayEquals(new double[] { 0.5, 0.5 }, label, 0.01);
//		System.out.println(Arrays.toString(label));
	}
	
	@Test
	public void testTrainingOneLayer() throws InputException {
		Network n = new Network(2, 1); // 2 input dimensions, one layer
		n.setLearningRate(0.1);
//		n.setDebugOn(true);
		n.configureLayer(1, new SigmoidNeuron[] { 
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {1,-2,-1,2})),
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {1,1,0,2})),
				});
		n.setLearningRate(1);
		double[] instance = new double[]{ 1, 1 };
		double[] label = new double[]{ 1, 0 };
		double[] classification = n.feedForward(instance);
		int i = 0; 
		while (i++ < 30) {
			n.train(instance, label);
//			classification = n.feedForward(instance);
//			System.out.println("Iteration " + i + " Classification: " + Arrays.toString(classification));
		}
		classification = n.feedForward(instance);
		assertArrayEquals(label, classification, 0.05);
	}

	@Test
	public void testTrainingTwoLayers() throws InputException {
		Network n = new Network(2, 2); // 2 input dimensions, one layer
		PrintStream oldPrintStream = System.out;
		// reset printstream to discard all output
		PrintStream discardPrintStream = new PrintStream(new DiscardOutputStream());
		System.setOut(discardPrintStream);
		n.setDebugOn(true);
		assertEquals(true, n.isDebugOn());
		n.setComputeDotGraph(true);
		assertEquals(true, n.isComputeDotGraph());
		n.setLearningRate(1);

		n.configureLayer(1, new SigmoidNeuron[] { 
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {1,-2,-1,2})),
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {1,1,0,2})),
				});
		n.configureLayer(2, new SigmoidNeuron[] { 
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {1,-2,-1,2})),
				});
		n.setLearningRate(1);
		double[] instance = new double[]{ 1, 1 };
		double[] label = new double[]{ 1};
		double[] classification = n.feedForward(instance);
		int i = 0; 
		while (i++ < 50) {
			n.train(instance, label);

		}
		classification = n.feedForward(instance);
		assertArrayEquals(label, classification, 0.02);
		System.setOut(oldPrintStream);
		discardPrintStream.close();
	}
	
	@Test(expected=NetworkLayerException.class)
	public void testLayerConfigurationExcessive() throws NetworkLayerException {
		Network n = new Network(1,1);
		n.configureLayer(3, null);
	}
	
	@Test(expected=NetworkLayerException.class)
	public void testLayerConfigurationReconfiguration() throws NetworkLayerException {
		Network n = new Network(1,1);
		n.configureLayer(1, null);
		n.configureLayer(1, null);
	}
	
}

