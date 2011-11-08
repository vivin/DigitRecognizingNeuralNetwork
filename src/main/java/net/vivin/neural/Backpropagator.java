package net.vivin.neural;

import java.util.List;
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

    public Backpropagator(NeuralNetwork neuralNetwork, double learningRate) {
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
    }

    public void train(double[][] inputs, double[][] expectedOutputs) {

        if(inputs.length != expectedOutputs.length) {
            throw new IllegalArgumentException("Number of inputs must equal number of outputs");
        }

        List<Layer> layers = neuralNetwork.getLayers();

        double totalError;

        do {

            totalError = 0;
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
                            //neuronError = output[k] * (1 - output[k]) * (expectedOutputs[k] - output[k]);
                        } else {
                            neuronError = neuron.getActivationStrategy().derivative(neuron.getOutput());
                            //neuronError = neuron.getOutput() * (1 - neuron.getOutput());

                            double sum = 0;
                            List<Neuron> downstreamNeurons = layer.getNextLayer().getNeurons();
                            for (Neuron downstreamNeuron : downstreamNeurons) {

                                int l = 0;
                                boolean found = false;
                                while (l < downstreamNeuron.getInputs().size() && !found) {
                                    Synapse synapse = downstreamNeuron.getInputs().get(l);

                                    if (synapse.getSourceNeuron() == neuron) {
                                        sum += (synapse.getWeight() * downstreamNeuron.getError());
                                    }

                                    l++;
                                }
                            }

                            neuronError *= sum;
                        }

                        neuron.setError(neuron.getError() + neuronError);

                        for (Synapse synapse : neuron.getInputs()) {
                            System.out.println("delta = " + learningRate + " * " + neuronError + " * " + synapse.getSourceNeuron().getOutput());
                            double delta = learningRate * neuronError * synapse.getSourceNeuron().getOutput();
                            System.out.println("Adjusting weight " + synapse.getWeight() + " by " + delta);
                            synapse.setWeight(synapse.getWeight() + delta);
                        }
                    }
                }

                output = neuralNetwork.getOutput();
                totalError += error(output, expectedOutput);
                System.out.println("inputs: " + explode(input) + " output: " + explode(output) + " expected: " + explode(expectedOutput) + " total error: " + totalError);
            }
        } while(totalError > 0.01);
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
