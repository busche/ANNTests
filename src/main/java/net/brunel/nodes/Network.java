package net.brunel.nodes;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import net.brunel.nodes.exceptions.InputDimensionMismatchException;
import net.brunel.nodes.exceptions.InputException;
import net.brunel.nodes.exceptions.NetworkLayerException;

public class Network {

	private static class InputNode implements Node {

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
		public double computeDerivative(double[] instanceData, int dimension) {
			return 0;
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
		public void updateW(double learningRate, int dimension, double gradientValue) {
			//noop
		}

		@Override
		public void updateB(double learningRate, double gradientValue) {
			//noop
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
			computeDotGraph(activations);
		}
		return activations[activations.length-1];		
	}

	private void computeDotGraph(double[][] temporaryValues) {
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
		
		System.out.println(sb.toString());		
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
		
		double[] classificationResult = feedForward(instanceData);
		double[] classificationError = new double[y.length];
		for (int i = 0; i < y.length; i++)
			classificationError[i] = classificationResult[i] - y[i];
		
		debug("Initial classification:       " + Arrays.toString(classificationResult));
		debug("Initial classification error: " + Arrays.toString(classificationError));
		
		backpropagate(y);
		
		updateWeights();
		
		debug("Errors:      " + Arrays.deepToString(errors));
	}
	
	private void updateWeights() {
		for (int l = numberOfLayers-1; l > 0 /*exclude input layer*/; l--) {
			Node[] currentNodes = nodesList.get(Integer.valueOf(l));
			Node[] previousNodes = nodesList.get(Integer.valueOf(l-1));
			
			for (int j = 0; j < currentNodes.length; j++) {

				double delta_b_j_l = errors[l][j];
				debug("Layer " + l + ", Node " + j + ", delta_b_j_l=" + delta_b_j_l);
				
				currentNodes[j].updateB(learningRate, delta_b_j_l);
				
				for (int k = 0; k < previousNodes.length; k++) {
					double delta_w_j_k_l = activations[l-1][k] * errors[l][j];
					debug("Layer " + l + ", Node " + k + ", delta_w_j_k_l=" + delta_w_j_k_l);
					currentNodes[j].updateW(learningRate, k, delta_w_j_k_l);
				}
				
			}
		}
	}
	
	private void backpropagate(double[] y) {
		
		computeErrorsOfLastLayer(y);

		// propagate errors backward
		int currentLayerIdx = numberOfLayers-1;
		int previousLayerIdx = numberOfLayers-2;
		Node[] currentLayer;
		Node[] previousLayer;

		while (currentLayerIdx>1)
		{
			currentLayerIdx=previousLayerIdx;
			previousLayerIdx = currentLayerIdx-1;
			int nextLayerIdx = currentLayerIdx+1;
			
			currentLayer = nodesList.get(Integer.valueOf(currentLayerIdx));
			previousLayer = nodesList.get(Integer.valueOf(previousLayerIdx));
				
			errors[currentLayerIdx] = new double[currentLayer.length];
	
			for (int j = 0; j < currentLayer.length; j++) {
				// right part
				double z_l_L =0;
				Node currentNode = currentLayer[j];
				for (int k = 0; k < previousLayer.length; k++)
					z_l_L += currentNode.w(k)*activations[previousLayerIdx][k];
				z_l_L += currentNode.b();
				
				// left part
				double errorContribution = 0;
				Node[] nextLayersNodes = nodesList.get(Integer.valueOf(nextLayerIdx));
				for (int n = 0; n < nextLayersNodes.length; n++) {
					double w = nextLayersNodes[n].w(j);
					double e = errors[nextLayerIdx][n];
					errorContribution += w*e;
				}
				errors[currentLayerIdx][j] = z_l_L*errorContribution;
				debug("Layer " + currentLayerIdx + ", Node " + j + ", z_l_L=" + z_l_L + " errorContribution=" + errorContribution );
			}
		}
	}
	
	private void computeErrorsOfLastLayer(double[] y) {
		int currentLayerIdx = numberOfLayers-1;
		int previousLayerIdx = numberOfLayers-2;
		Node[] currentLayer = nodesList.get(Integer.valueOf(currentLayerIdx));
		Node[] previousLayer = nodesList.get(Integer.valueOf(previousLayerIdx));
		
		if (errors[currentLayerIdx] != null) {
			if (errors[currentLayerIdx].length == currentLayer.length) {
				// all fine.
			} else {
				errors[currentLayerIdx]=null;
				System.gc();
				errors[currentLayerIdx] = new double[currentLayer.length];
			}
		} else {
			errors[currentLayerIdx] = new double[currentLayer.length];
		}

		for (int j = 0; j < currentLayer.length; j++) {
			double z_j_L =0;
			Node currentNode = currentLayer[j];
			for (int k = 0; k < previousLayer.length; k++)
				z_j_L += currentNode.w(k)*activations[previousLayerIdx][k];
			z_j_L += currentNode.b();
			
			double sigmoidPrime = MyMath.sigmoid(z_j_L)*(1-MyMath.sigmoid(z_j_L));
			double a_j_L = MyMath.sigmoid(z_j_L);
			double deltaC_vs_deltaA_j_L = (a_j_L - y[j]);
			double error = deltaC_vs_deltaA_j_L*sigmoidPrime;
			errors[currentLayerIdx][j] = error;
		}
	}

}
