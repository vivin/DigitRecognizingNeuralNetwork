package net.vivin.neural;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 1:39 PM
 */
public class NeuralNetwork {

    private List<Layer> layers;
    private Layer input;
    private Layer output;

    public NeuralNetwork() {
        layers = new ArrayList<Layer>();
    }

    public void addLayer(Layer layer) {
        layers.add(layer);

        if(layers.size() == 1) {
            input = layer;
        }

        output = layers.get(layers.size() - 1);
    }

    public void setInputs(double[] inputs) {
        if(input != null) {
            if(input.getNeurons().size() != inputs.length) {
                throw new IllegalArgumentException("The number of inputs must equal the number of neurons in the input layer");
            }

            else {
                int i = 0;
                for(Neuron neuron : input.getNeurons()) {
                    neuron.setOutput(inputs[i]);
                    i++;
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
}
