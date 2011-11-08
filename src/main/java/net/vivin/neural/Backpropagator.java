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

    public Backpropagator(NeuralNetwork neuralNetwork, double learningRate) {
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
    }

    public void train(double[][] inputs, double[][] expectedOutputs) {

        if(inputs.length != expectedOutputs.length) {
            throw new IllegalArgumentException("Number of inputs must equal number of outputs");
        }

        List<Layer> layers = neuralNetwork.getLayers();

        Map<Neuron, Double> neuronDeltaMap = new HashMap<Neuron, Double>();

        double totalError;

        do {

            totalError = 0;
            double[] errors = new double[inputs.length];

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
                                    }

                                    l++;
                                }
                            }

                            neuronError *= sum;
                        }

                        //neuron.setError(neuron.getError() + neuronError);
                        neuron.setError(neuronError);

                        for (Synapse synapse : neuron.getInputs()) {
                            //System.out.println("delta = " + learningRate + " * " + neuronError + " * " + synapse.getSourceNeuron().getOutput());
                            double delta = learningRate * neuronError * synapse.getSourceNeuron().getOutput();

                            if(neuronDeltaMap.get(neuron) != null) {
                                double previous_delta = neuronDeltaMap.get(neuron);
                                delta += 0.75 * previous_delta;
                            }

                            neuronDeltaMap.put(neuron, delta); //read up more on this... figure out how to fix the algorithm to account for momentum

                            //System.out.println("Adjusting weight " + synapse.getWeight() + " by " + delta);
                            synapse.setWeight(synapse.getWeight() + delta);
                        }
                    }
                }

                output = neuralNetwork.getOutput();
                errors[i] = error(output, expectedOutput);
                //System.out.println("inputs: " + explode(input) + " output: " + explode(output) + " expected: " + explode(expectedOutput) + " error: " + errors[i]);
            }

            for(double error : errors) {
                totalError += error;
            }

            totalError = totalError / errors.length;

            System.out.println("Total error for this training set: " + totalError);

        } while(totalError > 0.001);
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
