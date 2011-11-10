package net.vivin.neural;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/6/11
 * Time: 12:19 PM
 */
public class Backpropagator {

    private NeuralNetwork neuralNetwork;
    private double learningRate;
    private double momentum;

    public Backpropagator(NeuralNetwork neuralNetwork, double learningRate, double momentum) {
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    public void train(int iterations, double[][] inputs, double[][] expectedOutputs) {

        if(inputs.length != expectedOutputs.length) {
            throw new IllegalArgumentException("Number of inputs must equal number of outputs");
        }

        double error = 100;
        int bestIteration = -1;
        NeuralNetwork best = neuralNetwork;

        for(int i = 0; i < iterations; i++) {

            System.out.println("\nIteration #" + i);

            double currentError = backpropagate(inputs, expectedOutputs);

            if(currentError < error) {
                best = neuralNetwork.copy();
                bestIteration = i;
                error = currentError;
            }

            neuralNetwork.reset();
        }

        neuralNetwork.copyWeightsFrom(best);
        System.out.println("\nBest error was " + error + " in iteration #" + bestIteration + "\n");
    }

    private double backpropagate(double[][] inputs, double[][] expectedOutputs) {
        List<Layer> layers = neuralNetwork.getLayers();

        Map<Synapse, Double> synapseNeuronDeltaMap = new HashMap<Synapse, Double>();

        double error;
        int epoch = 1;

        do {

            error = 0;

            for (int i = 0; i < inputs.length; i++) {

                double[] input = inputs[i];
                double[] expectedOutput = expectedOutputs[i];

                neuralNetwork.setInputs(input);
                double[] output = neuralNetwork.getOutput();

                for (int j = layers.size() - 1; j > 0; j--) {
                    Layer layer = layers.get(j);

                    for (int k = 0; k < layer.getNeurons().size(); k++) {
                        Neuron neuron = layer.getNeurons().get(k);
                        double neuronError = 0;

                        if (layer.isOutputLayer()) {
                            neuronError = neuron.getActivationStrategy().derivative(output[k]) * (expectedOutput[k] - output[k]);
                        } else {
                            neuronError = neuron.getActivationStrategy().derivative(neuron.getOutput());

                            double sum = 0;
                            List<Neuron> downstreamNeurons = layer.getNextLayer().getNeurons();
                            for (Neuron downstreamNeuron : downstreamNeurons) {

                                int l = 0;
                                boolean found = false;
                                while (l < downstreamNeuron.getInputs().size() && !found) {
                                    Synapse synapse = downstreamNeuron.getInputs().get(l);

                                    if (synapse.getSourceNeuron() == neuron) {
                                        sum += (synapse.getWeight() * downstreamNeuron.getError());
                                        found = true;
                                    }

                                    l++;
                                }
                            }

                            neuronError *= sum;
                        }

                        neuron.setError(neuronError);

                        for (Synapse synapse : neuron.getInputs()) {
                            //System.out.println("delta = " + learningRate + " * " + neuronError + " * " + synapse.getSourceNeuron().getOutput());
                            double delta = learningRate * neuronError * synapse.getSourceNeuron().getOutput();

                            if(synapseNeuronDeltaMap.get(synapse) != null) {
                                double previous_delta = synapseNeuronDeltaMap.get(synapse);
                                delta += momentum * previous_delta;
                            }

                            synapseNeuronDeltaMap.put(synapse, delta); //read up more on this... figure out how to fix the algorithm to account for momentum

                            //System.out.println("Adjusting weight " + synapse.getWeight() + " by " + delta);
                            synapse.setWeight(synapse.getWeight() + delta);
                        }
                    }
                }

                output = neuralNetwork.getOutput();

                error += error(output, expectedOutput);
                //System.out.println("inputs: " + explode(input) + " output: " + explode(output) + " expected: " + explode(expectedOutput) + " error: " + errors[i]);
            }

            error /= 2;

            System.out.println("Error for epoch " + epoch + ": " + error);
            epoch++;

        } while(error > 0.001);
        return error;
    }

    private String explode(double[] array) {
        String string = "[";

        for (double number : array) {
            string += number + ", ";
        }

        Pattern pattern = Pattern.compile(", $", Pattern.DOTALL);
        Matcher matcher = pattern.matcher(string);
        string = matcher.replaceAll("");

        return string + "]";
    }

    public double error(double[] actual, double[] expected) {

        if (actual.length != expected.length) {
            throw new IllegalArgumentException("The lengths of the actual and expected value arrays must be equal");
        }

        double sum = 0;

        for (int i = 0; i < expected.length; i++) {
            sum += Math.pow(expected[i] - actual[i], 2);
        }

        return sum;
    }
}
