package net.brunel.nodes;

import static org.junit.Assert.*;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;

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
		while (i++ < 150) {
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
		while (i++ < 1000) {
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
	
	@Test
	public void testBatchTrainingOneLayerOneOutput() throws InputException {
		Network n = new Network(2, 2); // 2 input dimensions, one layer
//		n.setDebugOn(true);
//		n.setComputeDotGraph(true);
		n.configureLayer(1, new SigmoidNeuron[] { 
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {1,0,1})),
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {0,1,-1})),
				});
		n.configureLayer(2, new SigmoidNeuron[] { 
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {0.3,0.7,0})),
				});
		double learningRate = 1;
		n.setLearningRate(learningRate);
		double[][] instances = new double[][] {
			new double[]{ 1, 0 },
			new double[]{ 1, 1 },
			new double[]{ 2, 1 },
//			new double[]{ 2, 1 },
		};
		double[][] labels = new double[][]{
			new double[]{ 1 },
			new double[]{ 0},
			new double[]{ 1 },
//			new double[]{ 1, -1 },
		};
		
		n.setLearningRateMultiplier(100, 0.995);
		n.trainBatch(instances, labels, 5000);
		
		double[] classification;
		for (int j = 0; j < instances.length; j++) {
//			classification = n.dumpDotGraph(instances[j], System.out);
			classification = n.feedForward(instances[j]);
			assertArrayEquals(labels[j], classification, 0.05);
		}
	}
	

	@Test
	public void testBatchTrainingOneLayerTwoOutputs() throws InputException {
		Network n = new Network(2, 2); // 2 input dimensions, two layer
		n.configureLayer(1, new SigmoidNeuron[] { 
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {1,-1,0})),
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {0,-1,0})),
				});
		n.configureLayer(2, new SigmoidNeuron[] { 
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {2,-2,0})),
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {-1,1,0})),
				});

		double learningRate = 0.75;
		n.setLearningRate(learningRate);
		
		double[][] instances = new double[][] {
			new double[]{ 1, 0 },
			new double[]{ 0, 1 },
			new double[]{ -1, -1 },
		};
		double[][] labels = new double[][]{
			new double[]{ 1,0 },
			new double[]{ 1,0},
			new double[]{ 0,1 },
		};
		
		int i = 0; 
		double[] classification;
		n.setIntelligentLearningRate(true);
		n.trainBatch(instances, labels, 50);

//		classification = n.feedForward(instances[0]);
//		System.out.print("Iteration " + i + " Classification 0: " + Arrays.toString(classification));
//		classification = n.feedForward(instances[1]);
//		System.out.print(" C 1: " + Arrays.toString(classification));
//		classification = n.feedForward(instances[2]);
//		System.out.print(" C 2: " + Arrays.toString(classification));
//		System.out.println();

		for (int j = 0; j < instances.length; j++) {
			classification = n.feedForward(instances[j]);
			// this is needed as the labels do not really approach 0 and 1, but sth. likt 0.87 and 0.12
			n.discretize(classification);
			assertArrayEquals(labels[j], classification, 0.05);
		}
	}
	
	@Test
	public void testTrainSingleNode() throws InputException {
		Network n = new Network(1, 2); // input dimension, layer
//		n.setDebugOn(true);
//		n.setComputeDotGraph(true);
		n.configureLayer(1, new SigmoidNeuron[] { 
//				new SigmoidNeuron(1, InitializerHelper.newCircularInitializer(new double[] {0.6,0.9,0})),
				new SigmoidNeuron(1, InitializerHelper.newCircularInitializer(new double[] {-0.6,-0.9})),
				});
		n.configureLayer(2, new SigmoidNeuron[] { 
//				new SigmoidNeuron(1, InitializerHelper.newCircularInitializer(new double[] {0.6,0.9,0})),
				new SigmoidNeuron(1, InitializerHelper.newCircularInitializer(new double[] {0.6, 0.9,0})),
				});

		double learningRate = 0.75;
		n.setLearningRate(learningRate);
		double[][] instances = new double[][] {
			new double[]{ 1},
//			new double[]{ 0},
		};
		double[][] labels = new double[][]{
			new double[]{ 0},
//			new double[]{ 1},
		};
		
		double[] classification;
		n.setLearningRateMultiplier(10000, 1);
//		n.setPrintWeights(true);
		
		n.trainBatch(instances, labels, 350);
		
		for (int j = 0; j < instances.length; j++) {
			classification = n.feedForward(instances[j]);
			
			assertArrayEquals(labels[j], classification, 0.05);
		}
	}
	
	@Test
	public void testCrossEntropyTrainSingleNode() throws InputException {
		Network n = new Network(1, 1); // input dimension, layer
//		n.setDebugOn(true);
//		n.setComputeDotGraph(true);
		n.configureLayer(1, new SigmoidNeuron[] { 
//				new SigmoidNeuron(1, InitializerHelper.newCircularInitializer(new double[] {0.6,0.9,0})),
				new SigmoidNeuron(1, InitializerHelper.newCircularInitializer(new double[] {3.0,3.0})),
				});

		double learningRate = 0.15;
		n.setLearningRate(learningRate);
		double[][] instances = new double[][] {
			new double[]{ 1+Math.random()*0.3},
			new double[]{ 1+Math.random()*0.3},
			new double[]{ 1+Math.random()*0.3},
			new double[]{ 1+Math.random()*0.3},
			new double[]{ 1+Math.random()*0.3},
			new double[]{ 1+Math.random()*0.3},
			new double[]{ 1+Math.random()*0.3},
			new double[]{ 1+Math.random()*0.3},
			new double[]{ 1+Math.random()*0.3},
//			new double[]{ 0},
		};
		double[][] labels = new double[][]{
			new double[]{ 0},
			new double[]{ 0},
			new double[]{ 0},
			new double[]{ 0},
			new double[]{ 0},
			new double[]{ 0},
			new double[]{ 0},
			new double[]{ 0},
			new double[]{ 0},
//			new double[]{ 1},
		};
		
		int i = 0; 
		double[] classification;
		n.setLearningRateMultiplier(10000, 1);
		n.setLossFunction(LossFunctionHelper.CROSS_ENTROPY_LOSS);
//		n.setDebugOn(true);
		
//		n.setPrintWeights(true);
		
		while (i++ < 3000) {
			try {
				n.trainIterationBatch(instances, labels);
			} catch (IterationException e) {
				System.out.println("Stopping iterations at iteration " + i + ", cannot reduce error any further!");
				break;
			}
			if (i % 100 == 0) {
				classification = n.feedForward(instances[0]);
				System.out.println(Arrays.toString(classification));

			}
		}		
		for (int j = 0; j < instances.length; j++) {
			classification = n.feedForward(instances[j]);
			
			assertArrayEquals(labels[j], classification, 0.05);
		}
	}
	
	@Test
	public void testMiniDatasetOneLayerOneOutput() throws InputException {
		Network n = new Network(2, 1); // 2 input dimensions, two layer
		n.configureLayer(1, new SigmoidNeuron[] { 
				new SigmoidNeuron(2, InitializerHelper.newCircularInitializer(new double[] {1,-1,0.5})),
				});

		double learningRate = 10;
		n.setLearningRate(learningRate);

		double[][] instances = new double[50][];
		double[][] labels = new double[50][];
		createDiagonalData(instances, labels);
		
		int i = 0; 
		double[] classification;
//		n.setIntelligentLearningRate(true);
//		n.setPrintWeights(true);
//		n.setLossFunction(LossFunctionHelper.CROSS_ENTROPY_LOSS);

		n.trainBatch(instances, labels, 750);
		
		for (int j = 0; j < instances.length; j++) {
			classification = n.feedForward(instances[j]);
			// this is needed as the labels do not really approach 0 and 1, but sth. likt 0.87 and 0.12
			n.discretize(classification);
			assertArrayEquals(labels[j], classification, 0.05);
		}
	}

	private void createDiagonalData(double[][] instances, double[][] labels) {
		Random r = new Random(100);
		
		for (int i = 0; i < instances.length; i++) {
			instances[i] = new double[2];
			labels[i] = new double[1];

			// x
			instances[i][0] = r.nextGaussian();
			// y
			instances[i][1] = -1 * r.nextGaussian();
			
			labels[i][0] = instances[i][0]+instances[i][1];
			labels[i][0] = (labels[i][0]>0?1:0);
//			System.out.println(String.format("%1$f %2$f %3$f", instances[i][0], instances[i][1], labels[i][0]));
		}
	}
	
	@Test
	public void testTrainSingleNodeWithRegularization() throws InputException {
		Network n = new Network(1, 2); // input dimension, layer
//		n.setDebugOn(true);
//		n.setComputeDotGraph(true);
		SigmoidNeuron[] layer1 = new SigmoidNeuron[] { 
				new SigmoidNeuron(1, InitializerHelper.newCircularInitializer(new double[] {-0.6,-0.9})),
				};
		SigmoidNeuron[] layer2 = new SigmoidNeuron[] { 
				new SigmoidNeuron(1, InitializerHelper.newCircularInitializer(new double[] {0.6, 0.9,0})),
				};
		double lambda=0.1;
		double learningRate = 0.05;

		layer1[0].setLambda(lambda);
		layer2[0].setLambda(lambda);
		n.configureLayer(1, layer1);
		n.configureLayer(2, layer2);

		n.setLearningRate(learningRate);
		double[][] instances = new double[][] {
			new double[]{ 1},
		};
		double[][] labels = new double[][]{
			new double[]{ 0},
		};
		
		double[] classification;
		n.setPrintWeights(true);
		
		int i=0;
		while (i++ < 6000) {
			try {
				n.trainIterationBatch(instances, labels);
			} catch (IterationException e) {
				System.out.println("Stopping iterations at iteration " + i + ", cannot reduce error any further!");
				break;
			}
			
			if (i % 10 == 0) {
				classification = n.feedForward(instances[0]);
//				System.out.println("Classification:" + Arrays.toString(classification));
				System.out.println("Error:          " + n.computeError(instances, labels));
			}
			
		}
		
		for (int j = 0; j < instances.length; j++) {
			classification = n.feedForward(instances[j]);
			
			assertArrayEquals(labels[j], classification, 0.05);
		}
	}
}

