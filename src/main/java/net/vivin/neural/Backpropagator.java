package net.vivin.neural;

import net.vivin.neural.generator.TrainingData;
import net.vivin.neural.generator.TrainingDataGenerator;

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
    private double characteristicTime;
    private double currentEpoch;

    public Backpropagator(NeuralNetwork neuralNetwork, double learningRate, double momentum, double characteristicTime) {
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.characteristicTime = characteristicTime;
    }

    public void train(TrainingDataGenerator generator, double errorThreshold) {

        double error;
        double sum = 0.0;
        double average = 25;
        int epoch = 1;
        int samples = 25;
        double[] errors = new double[samples];

        do {
            TrainingData trainingData = generator.getTrainingData();
            error = backpropagate(trainingData.getInputs(), trainingData.getOutputs());

            sum -= errors[epoch % samples];
            errors[epoch % samples] = error;
            sum += errors[epoch % samples];

            if(epoch > samples) {
                average = sum / samples;
            }

            System.out.println("Error for epoch " + epoch + ": " + error + ". Average: " + average + (characteristicTime > 0 ? " Learning rate: " + learningRate / (1 + (currentEpoch / characteristicTime)): ""));
            epoch++;
            currentEpoch = epoch;
        } while(average > errorThreshold);
    }

    public double backpropagate(double[][] inputs, double[][] expectedOutputs) {

        double error = 0;

        Map<Synapse, Double> synapseNeuronDeltaMap = new HashMap<Synapse, Double>();

        for (int i = 0; i < inputs.length; i++) {

            double[] input = inputs[i];
            double[] expectedOutput = expectedOutputs[i];

            List<Layer> layers = neuralNetwork.getLayers();

            neuralNetwork.setInputs(input);
            double[] output = neuralNetwork.getOutput();

            //First step of the backpropagation algorithm. Backpropagate errors from the output layer all the way up
            //to the first hidden layer
            for (int j = layers.size() - 1; j > 0; j--) {
                Layer layer = layers.get(j);

                for (int k = 0; k < layer.getNeurons().size(); k++) {
                    Neuron neuron = layer.getNeurons().get(k);
                    double neuronError = 0;

                    if (layer.isOutputLayer()) {
                        //the order of output and expected determines the sign of the delta. if we have output - expected, we subtract the delta
                        //if we have expected - output we add the delta.
                        neuronError = neuron.getDerivative() * (output[k] - expectedOutput[k]);
                    } else {
                        neuronError = neuron.getDerivative();

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
                }
            }

            //Second step of the backpropagation algorithm. Using the errors calculated above, update the weights of the
            //network
            for(int j = layers.size() - 1; j > 0; j--) {
                Layer layer = layers.get(j);

                for(Neuron neuron : layer.getNeurons()) {

                    for(Synapse synapse : neuron.getInputs()) {

                        double newLearningRate = characteristicTime > 0 ? learningRate / (1 + (currentEpoch / characteristicTime)) : learningRate;
                        double delta = newLearningRate * neuron.getError() * synapse.getSourceNeuron().getOutput();

                        if(synapseNeuronDeltaMap.get(synapse) != null) {
                            double previousDelta = synapseNeuronDeltaMap.get(synapse);
                            delta += momentum * previousDelta;
                        }

                        synapseNeuronDeltaMap.put(synapse, delta);
                        synapse.setWeight(synapse.getWeight() - delta);
                    }
                }
            }

            output = neuralNetwork.getOutput();
            error += error(output, expectedOutput);
        }

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

        return sum / 2;
    }
}