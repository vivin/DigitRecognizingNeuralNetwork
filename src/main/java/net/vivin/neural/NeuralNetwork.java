package net.vivin.neural;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 1:39 PM
 */
public class NeuralNetwork implements Serializable {

    private List<Layer> layers;
    private Layer input;
    private Layer output;

    public NeuralNetwork() {
        layers = new ArrayList<Layer>();
    }

    public NeuralNetwork copy() {
        NeuralNetwork copy = new NeuralNetwork();

        Layer previousLayer = null;
        for(Layer layer : layers) {

            Layer layerCopy;

            if(layer.hasBias()) {
                Neuron bias = layer.getNeurons().get(0);
                Neuron biasCopy = new Neuron(bias.getActivationStrategy().copy());
                biasCopy.setOutput(bias.getOutput());
                layerCopy = new Layer(null, biasCopy);
            }

            else {
                layerCopy = new Layer();
            }

            layerCopy.setPreviousLayer(previousLayer);

            int biasCount = layerCopy.hasBias() ? 1 : 0;

            for(int i = biasCount; i < layer.getNeurons().size(); i++) {
                Neuron neuron = layer.getNeurons().get(i);

                Neuron neuronCopy = new Neuron(neuron.getActivationStrategy().copy());
                neuronCopy.setOutput(neuron.getOutput());
                neuronCopy.setError(neuron.getError());

                if(neuron.getInputs().size() == 0) {
                    layerCopy.addNeuron(neuronCopy);
                }

                else {
                    double[] weights = neuron.getWeights();
                    layerCopy.addNeuron(neuronCopy, weights);
                }
            }

            copy.addLayer(layerCopy);
            previousLayer = layerCopy;
        }

        return copy;
    }

    public void addLayer(Layer layer) {
        layers.add(layer);

        if(layers.size() == 1) {
            input = layer;
        }

        if(layers.size() > 1) {
            //clear the output flag on the previous output layer, but only if we have more than 1 layer
            Layer previousLayer = layers.get(layers.size() - 2);
            previousLayer.setNextLayer(layer);
        }

        output = layers.get(layers.size() - 1);
    }

    public void setInputs(double[] inputs) {
        if(input != null) {

            int biasCount = input.hasBias() ? 1 : 0;

            if(input.getNeurons().size() - biasCount != inputs.length) {
                throw new IllegalArgumentException("The number of inputs must equal the number of neurons in the input layer");
            }

            else {
                List<Neuron> neurons = input.getNeurons();
                for(int i = biasCount; i < neurons.size(); i++) {
                    neurons.get(i).setOutput(inputs[i - biasCount]);
                }
            }
        }
    }

    public double[] getOutput() {

        double[] outputs = new double[output.getNeurons().size()];

        for(int i = 1; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            layer.feedForward();
        }

        int i = 0;
        for(Neuron neuron : output.getNeurons()) {
            outputs[i] = neuron.getOutput();
            i++;
        }

        return outputs;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void reset() {
        for(Layer layer : layers) {
            for(Neuron neuron : layer.getNeurons()) {
                for(Synapse synapse : neuron.getInputs()) {
                    synapse.setWeight((Math.random() * 1) - 0.5);
                }
            }
        }
    }

    public void copyWeightsFrom(NeuralNetwork sourceNeuralNetwork) {
        if(layers.size() != sourceNeuralNetwork.layers.size()) {
            throw new IllegalArgumentException("Cannot copy weights. Number of layers do not match (" + sourceNeuralNetwork.layers.size() + " in source versus " + layers.size() + " in destination)");
        }

        int i = 0;
        for(Layer sourceLayer : sourceNeuralNetwork.layers) {
            Layer destinationLayer = layers.get(i);

            if(destinationLayer.getNeurons().size() != sourceLayer.getNeurons().size()) {
                throw new IllegalArgumentException("Number of neurons do not match in layer " + (i + 1) + "(" + sourceLayer.getNeurons().size() + " in source versus " + destinationLayer.getNeurons().size() + " in destination)");
            }

            int j = 0;
            for(Neuron sourceNeuron : sourceLayer.getNeurons()) {
                Neuron destinationNeuron = destinationLayer.getNeurons().get(j);

                if(destinationNeuron.getInputs().size() != sourceNeuron.getInputs().size()) {
                    throw new IllegalArgumentException("Number of inputs to neuron " + (j + 1) + " in layer " + (i + 1) + " do not match (" + sourceNeuron.getInputs().size() + " in source versus " + destinationNeuron.getInputs().size() + " in destination)");
                }

                int k = 0;
                for(Synapse sourceSynapse : sourceNeuron.getInputs()) {
                    Synapse destinationSynapse = destinationNeuron.getInputs().get(k);

                    destinationSynapse.setWeight(sourceSynapse.getWeight());
                    k++;
                }

                j++;
            }

            i++;
        }
    }
}
