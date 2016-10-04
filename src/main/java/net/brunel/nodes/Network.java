package net.brunel.nodes;

import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import net.brunel.nodes.exceptions.InputDimensionMismatchException;
import net.brunel.nodes.exceptions.InputException;
import net.brunel.nodes.exceptions.NetworkLayerException;

public class Network {

	private static class InputNode implements Node, Function {

		private double value;

		public InputNode(double value) {
			this.value = value;
		}

		@Override
		public double computeOutput(double[] input) throws InputException {
			return value;
		}

		@Override
		public double getWeightFromInput(int c) {
			return 1;
		}

		@Override
		public double w(int k) {
			return 0;
		}

		@Override
		public double b() {
			return 0;
		}

		@Override
		public void updateW(int dimension, double gradientValue) {
			//noop
		}

		@Override
		public void updateB(double gradientValue) {
			//noop
		}

		@Override
		public Function getFunction() {
			return this;
		}

		@Override
		public double computeDerivative(double z_l_L) {
			return 0;
		}

		@Override
		public double computeAt(double z_j_L) {
			return z_j_L;
		}

		@Override
		public void prepareUpdate() {
		}

		@Override
		public void commitUpdate(double learningRate) {
		}

		@Override
		public void configureUpdate(double d) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void resetUpdate() {
			// TODO Auto-generated method stub
			
		}

	}

	private final int numberOfLayers;
	private final int numberOfInputs;

	private boolean debugOn;
	private Map<Integer, Node[]> nodesList;
	private boolean computeDotGraph;
	private double learningRate;

	/**
	 * indexed by [layer][node]
	 */
	private double[][] activations;
	
	/**
	 * indexed by [layer][node]
	 */
	private double[][] errors;
	private LossFunction lossFunction;
	private int learningRateIterationAmount = 100;
	private double learningRateIterationDecay = 0.995;
	private boolean intelligentLearningRate;

	private void debug(String string) {
		if (debugOn)
			System.out.println(string);
	}
	public Network(int inputDimension, int numberOfLayers) {
		this.numberOfInputs = inputDimension;
		this.numberOfLayers = numberOfLayers+1;
		debugOn=false;
		computeDotGraph=false;
		nodesList = new HashMap<>(this.numberOfLayers);
		activations = new double[this.numberOfLayers][];
		activations[0] = new double[inputDimension];
		errors = new double[this.numberOfLayers][];
		lossFunction = (a,b) -> {return a-b;};
		intelligentLearningRate=false;
		
		Node[] inputLayer = new Node[inputDimension];
		for (int i = 0; i < inputDimension; i++)
			inputLayer[i] = new InputNode(0);
		
		try {
			configureLayer(0, inputLayer);
		} catch (NetworkLayerException e) {
			e.printStackTrace();
		}
		
	}


	public void configureLayer(int layerNumber, Node[] nodes) throws NetworkLayerException {
		if (layerNumber > nodesList.size())
			throw new NetworkLayerException("Layer configuration exceeds configured number of Layers");
		if (nodesList.containsKey(Integer.valueOf(layerNumber)))
				throw new NetworkLayerException("Layer already configured.");
		nodesList.put(layerNumber, nodes);
	}

	public int getNumberOfInputs() {
		return numberOfInputs;
	}

	public int getNumberOfLayers() {
		return numberOfLayers;
	}

	public double[] feedForward(double[] input) throws InputException {
		if (numberOfInputs!=input.length)
			throw new InputDimensionMismatchException(numberOfInputs, input.length);
		
		debug("layer 0        Output: " + Arrays.toString(input));

		for (int i = 0; i < input.length; i++) {
			activations[0][i]=input[i];
		}
				
		// iterate through the layer
		for (int l = 1; l < numberOfLayers; l++) {
			debug("layer " + (l) + "        Input:  " + Arrays.toString(activations[l-1]));

			Node[] currentLayer = nodesList.get(Integer.valueOf(l));
			
			activations[l] = new double[currentLayer.length];
			for (int j = 0; j < currentLayer.length; j++) {
				activations[l][j] = currentLayer[j].computeOutput(activations[l-1]);
			}
			debug("layer " + (l) + "        output: " + Arrays.toString(activations[l]));
		}
		debug("output layer, output:  " + Arrays.toString(activations[activations.length-1]));
		
		if (computeDotGraph) {
			computeDotGraph();
		}
		return activations[activations.length-1];		
	}

	private void computeDotGraph() {
		computeDotGraph(System.out);
	}
	
	private void computeDotGraph(PrintStream out) {
		double[][] temporaryValues=activations;
		StringBuffer sb = new StringBuffer();
		sb.append("digraph ANN {");
		sb.append("graph [splines=true overlap=false labelangle=100]; ");
		//sb.append("node[shape=ellipse];");
		
		// configure nodes
		for(int l = 0; l < temporaryValues.length; l++) {
			int numberOfNodesInCurrentLayer = temporaryValues[l].length;
			for (int j = 0; j < numberOfNodesInCurrentLayer; j++) {
				double bias=nodesList.get(l)[j].b();
				sb.append(computeNodeName(l, j) + " [label=\"bias=" + bias + ", output=" + temporaryValues[l][j] + "\"];");
			}
		}

		// configure edges
		for (int l = 1; l < temporaryValues.length; l++) {
			for (int c = 0; c < temporaryValues[l-1].length; c++) {
				for (int n = 0; n < temporaryValues[l].length; n++) {
					Node[] layerNodes = nodesList.get(Integer.valueOf(l));
					String edgeLabel = String.format(Locale.ENGLISH, "%.2f", layerNodes[n].getWeightFromInput(c));
					sb.append(computeNodeName(l-1, c) + "->" + computeNodeName(l, n) + " [label=\"" + edgeLabel + "\"];");
				}
		}
		
		sb.append("");
		sb.append("");			}
		sb.append("}");
		
		out.println(sb.toString());		
	}

	private String computeNodeName(int layer, int nodeIdx) {
		return "nl" + layer + "n" + nodeIdx;
	}

	public boolean isDebugOn() {
		return debugOn;
	}


	public void setDebugOn(boolean debugOn) {
		this.debugOn = debugOn;
	}


	public boolean isComputeDotGraph() {
		return computeDotGraph;
	}


	public void setComputeDotGraph(boolean computeDotGraph) {
		this.computeDotGraph = computeDotGraph;
	}


	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	
	public void train(double[] instanceData, double[] y) throws InputException {
		prepareUpdate();
		
		double[] classificationResult = feedForward(instanceData);
		double[] classificationError = new double[y.length];
		for (int i = 0; i < y.length; i++)
			classificationError[i] = classificationResult[i] - y[i];
		
		debug("Initial classification:       " + Arrays.toString(classificationResult));
		debug("Initial classification error: " + Arrays.toString(classificationError));
		
		computeErrorsOfLastLayer(y);

		backpropagateError();
		
		updateWeights();
		
		commitUpdate(learningRate);
		
		debug("Errors:      " + Arrays.deepToString(errors));
	}
	
	/*
	 * updates the weights in the overall network based on the individual error contributions
	 * which are previously computed.
	 */
	private void updateWeights() {
		for (int l = numberOfLayers - 1; l > 0 /* exclude input layer */; l--) {
			Node[] currentNodes = nodesList.get(Integer.valueOf(l));
			Node[] previousNodes = nodesList.get(Integer.valueOf(l - 1));

			for (int j = 0; j < currentNodes.length; j++) {

				double delta_b_j_l = errors[l][j];
				debug("Layer " + l + ", Node " + j + ", delta_b_j_l=" + delta_b_j_l);

				currentNodes[j].updateB( delta_b_j_l);

				for (int k = 0; k < previousNodes.length; k++) {
					double delta_w_j_k_l = activations[l - 1][k] * errors[l][j];
					debug("Layer " + l + ", Node " + k + ", delta_w_j_k_l=delta_w_" + j + "_" + k + "_" + l + "=" + delta_w_j_k_l);
					currentNodes[j].updateW(k, delta_w_j_k_l);
				}

			}
		}
	}
	
	private void backpropagateError() throws InputException {
		
		// propagate errors backward
		int currentLayerIdx = numberOfLayers-1;
		int previousLayerIdx = numberOfLayers-2;
		
		while (currentLayerIdx>1)
		{
			currentLayerIdx=previousLayerIdx;
			previousLayerIdx = currentLayerIdx-1;
			int nextLayerIdx = currentLayerIdx+1;
			
			Node[] currentLayer = nodesList.get(Integer.valueOf(currentLayerIdx));
				
			initErrorArrayAtIndex(currentLayerIdx);
	
			// for each node in the current layer ...
			for (int j = 0; j < currentLayer.length; j++) {
				// ... compute the error based on the derivative of the current node (z_l_L) and 
				// the errorContribution this node makes at sucessive layers 
				
				// right part
				Node currentNode = currentLayer[j];
				double z_l_L = currentNode.computeOutput(activations[previousLayerIdx]);
						
				// left part
				double errorContribution = 0;
				Node[] nextLayersNodes = nodesList.get(Integer.valueOf(nextLayerIdx));
				
				for (int n = 0; n < nextLayersNodes.length; n++) {
					// the impact of this nodes output (the weight from this node to node n in the next layer
					double w = nextLayersNodes[n].w(j);
					// the error at node n in the next layer
					double e = errors[nextLayerIdx][n];
					/*
					 * the errorContribution of this node is small if either our weight for the next layer is small,
					 * or if the overall error at the successive layer is small. 
					 */
					errorContribution += w*e;
				}
				/*
				 * we set "our" error to be the derivative of the current nodes activation times 
				 * our errorContribution to the next layer
				 */
				errors[currentLayerIdx][j] = currentNode.getFunction().computeDerivative(z_l_L)*errorContribution;
				
				debug("Layer " + currentLayerIdx + ", Node " + j + ", z_l_L=" + z_l_L + " errorContribution=" + errorContribution );
			}
		}
	}

	private void initErrorArrayAtIndex(int currentLayerIdx) {
		Node[] currentNodes = nodesList.get(Integer.valueOf(currentLayerIdx));

		if (errors[currentLayerIdx] != null) {
			if (errors[currentLayerIdx].length == currentNodes.length) {
				// all fine.
			} else {
				errors[currentLayerIdx] = null;
				System.gc();
				errors[currentLayerIdx] = new double[currentNodes.length];
			}
		} else {
			errors[currentLayerIdx] = new double[currentNodes.length];
		}
	}
	
	private void computeErrorsOfLastLayer(double[] y) throws InputException {
		int currentLayerIdx = numberOfLayers-1;
		int previousLayerIdx = numberOfLayers-2;
		Node[] currentLayer = nodesList.get(Integer.valueOf(currentLayerIdx));
		
		initErrorArrayAtIndex(currentLayerIdx);

		for (int j = 0; j < currentLayer.length; j++) {
			double z_j_L =0;

			Node currentNode = currentLayer[j];
//			for (int k = 0; k < previousLayer.length; k++)
//				z_j_L += currentNode.w(k)*activations[previousLayerIdx][k];
//			z_j_L += currentNode.b();
			z_j_L = currentNode.computeOutput(activations[previousLayerIdx]);
			
			double sigmoidPrime = currentNode.getFunction().computeDerivative(z_j_L);
			double a_j_L =  currentNode.getFunction().computeAt(z_j_L);
			double deltaC_vs_deltaA_j_L = lossFunction.computeLoss(a_j_L, y[j]);
			double error = deltaC_vs_deltaA_j_L*sigmoidPrime;
			errors[currentLayerIdx][j] = error;
		}
	}
	
	public void trainIterationBatch(double[][] instances, double[][] labels) throws InputException, IterationException {
		prepareUpdate();
		double[] iterationErrors = new double[labels[0].length];
		for (int i = 0; i < instances.length; i++) {
			double[] predictedLabelDistribution = feedForward(instances[i]);
//			System.out.println("trainBatch, Instance " + i + " predicted label distribution: " + Arrays.toString(predictedLabelDistribution) + " actual label distribution " + Arrays.toString(labels[i]));
		
			for (int j = 0; j < labels[i].length; j++)
				iterationErrors[j] += (labels[i][j]-predictedLabelDistribution[j])*(labels[i][j]-predictedLabelDistribution[j]);
			
			computeErrorsOfLastLayer(labels[i]);
			
			backpropagateError();

//			printErrors();
			
			updateWeights();
			
//			System.out.println("Printing update_weights after instance " + i);
//			printWeights();
		}
		double iterationErrorSum = 0;
		for (double d : iterationErrors)
			iterationErrorSum+=d;
		System.out.println("current iterationErrorSum = " + iterationErrorSum);
		
//		System.out.print("Press any key for next iteration ...");
//		try {
//			System.in.read();
//			System.out.println();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
		
		if (intelligentLearningRate) {
			
			
			double[] learningRates = new double[] {
	//				(10*learningRate) / (instances.length* 1),
					learningRate / (instances.length* 1),
	//				learningRate / (instances.length* 25),
					learningRate / (instances.length* 10),
					learningRate / (instances.length* 100),
	//				learningRate / (instances.length* 1000),
	//				learningRate / (instances.length* 10000),
			};
			double[] localLearningRateErrors = new double[learningRates.length];
			double minError = Double.MAX_VALUE;
			double bestLearningRate = 0;
			for (int i = 0; i < learningRates.length; i++) {
				configureUpdate(learningRates[i]);
				localLearningRateErrors[i] = computeError(instances, labels);
				
	//			System.out.println("Error for i=" + i + ": " + localLearningRateErrors[i]);
				if (localLearningRateErrors[i] < minError) {
					minError=localLearningRateErrors[i];
					bestLearningRate=learningRates[i];
				}
			}
			System.out.println("Best error is " + minError + " in array " + Arrays.toString(localLearningRateErrors));
			if (iterationErrorSum < minError) {
				System.out.println("No way out! Cannot reduce error!");
				resetUpdate();
				throw new IterationException("Cannot reduce error any further!");
			} 
			commitUpdate(bestLearningRate);
		} else {
			commitUpdate(learningRate);
			
		}
	}
	
	public double computeError(double[][] instances, double[][] labels) throws InputException {
		double[] iterationErrors = new double[labels[0].length];
		for (int i = 0; i < instances.length; i++) {
			double[] predictedLabelDistribution = feedForward(instances[i]);
			for (int j = 0; j < labels[i].length; j++)
				iterationErrors[j] += (labels[i][j] - predictedLabelDistribution[j])
						* (labels[i][j] - predictedLabelDistribution[j]);
			debug("iterationErrors = " + Arrays.toString(iterationErrors));
		}
		double iterationErrorSum = 0;
		for (double d : iterationErrors)
			iterationErrorSum += d;
		debug("iterationErrorSum = " + iterationErrorSum);
		return iterationErrorSum;

	}
	
	private void printWeights() {

		for (int l = 1; l < numberOfLayers /* exclude input layer */; l++) {
			Node[] currentNodes = nodesList.get(Integer.valueOf(l));
			for(int k = 0; k < currentNodes.length; k++) {
				System.out.print("layer_" + l + "_node_" + k + "_" + Arrays.toString(((SigmoidNeuron) currentNodes[k]).updateWeights) + " ");				
			}
			System.out.println();
		}
		System.out.println();

	}
	private void printErrors() {
		System.out.print("Printing errors ");

		for (double[] error:errors)
			System.out.print(Arrays.toString(error) + "--");
		
		System.out.println();
	}
	private void prepareUpdate() {
		for (Node[] n1 : nodesList.values())
			for (Node n : n1)
				n.prepareUpdate();
	}

	private void configureUpdate(double myLearningRate) {
		for (Node[] n1 : nodesList.values())
			for (Node n : n1)
				n.configureUpdate(myLearningRate);
	}

	private void commitUpdate(double myLearningRate) {
		for (Node[] n1 : nodesList.values())
			for (Node n : n1)
				n.commitUpdate(myLearningRate);
	}

	
	private void resetUpdate() {
		for (Node[] n1 : nodesList.values())
			for (Node n : n1)
				n.resetUpdate();
	}
	
	public double[] dumpDotGraph(double[] instance, PrintStream out) throws InputException {
		double[] classification = feedForward(instance);
		computeDotGraph(out);
		return classification;
	}
	
	public void trainBatch(double[][] instances, double[][] labels, int numIterations) throws InputException {
	
		int i=0;
		while (i++ < numIterations) {
			try {
				trainIterationBatch(instances, labels);
			} catch (IterationException e) {
				System.out.println("Stopping iterations at iteration " + i + ", cannot reduce error any further!");
				break;
			}
			
			if (i % learningRateIterationAmount  == 0) {
				learningRate *=learningRateIterationDecay;
				debug("LearningRate is now " + learningRate);
				
				setLearningRate(learningRate);
			}
		}

	}
	public void setLearningRateMultiplier(int iterationNumber, double learningRateIterationDecay) {
		this.learningRateIterationAmount = iterationNumber;
		this.learningRateIterationDecay = learningRateIterationDecay;
	}
	public int getLearningRateIterationAmount() {
		return learningRateIterationAmount;
	}
	public double getLearningRateIterationDecay() {
		return learningRateIterationDecay;
	}
	public double getLearningRate() {
		return learningRate;
	}

}
