package net.vivin.neural;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 12:42 PM
 */
public class Layer {

    private List<Neuron> neurons;
    private Layer previousLayer;

    public Layer() {
        neurons = new ArrayList<Neuron>();
        previousLayer = null;
    }

    public Layer(Layer previousLayer) {
        this();
        this.previousLayer = previousLayer;
    }

    public List<Neuron> getNeurons() {
        return this.neurons;
    }

    public void addNeuron(Neuron neuron) {

        neurons.add(neuron);

        if(previousLayer != null) {
            for(Neuron previousLayerNeuron : previousLayer.getNeurons()) {
                neuron.addInput(new Synapse(previousLayerNeuron, 0.0));
            }
        }
    }

    public void addNeuron(Neuron neuron, double[] weights) {

        neurons.add(neuron);

        if(previousLayer != null) {

            if(previousLayer.getNeurons().size() != weights.length) {
                throw new IllegalArgumentException("The number of weights supplied must be equal to the number of neurons in the previous layer");
            }

            else {
                int i = 0;
                for(Neuron previousLayerNeuron : previousLayer.getNeurons()) {
                    neuron.addInput(new Synapse(previousLayerNeuron, weights[i]));
                    i++;
                }
            }

        }
    }

    public void feedForward() {

        for(Neuron neuron : neurons) {
            double weightedSum = 0.0;

            for(Synapse input : neuron.getInputs()) {
                weightedSum += (input.getWeight() * input.getSourceNeuron().getOutput());
            }

            neuron.setOutput(neuron.getActivationStrategy().activate(weightedSum));
        }
    }
}
