package net.vivin.neural;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 12:42 PM
 */
public class Layer implements Serializable {

    private List<Neuron> neurons;
    private Layer previousLayer;
    private Layer nextLayer;
    private Neuron bias;

    public Layer() {
        neurons = new ArrayList<Neuron>();
        previousLayer = null;
    }

    public Layer(Layer previousLayer) {
        this();
        this.previousLayer = previousLayer;
    }

    public Layer(Layer previousLayer, Neuron bias) {
        this(previousLayer);
        this.bias = bias;
        neurons.add(bias);
    }

    public List<Neuron> getNeurons() {
        return this.neurons;
    }

    public void addNeuron(Neuron neuron) {

        neurons.add(neuron);

        if(previousLayer != null) {
            for(Neuron previousLayerNeuron : previousLayer.getNeurons()) {
                neuron.addInput(new Synapse(previousLayerNeuron, (Math.random() * 1) - 0.5)); //initialize with a random weight between -1 and 1
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
                List<Neuron> previousLayerNeurons = previousLayer.getNeurons();
                for(int i = 0; i < previousLayerNeurons.size(); i++) {
                    neuron.addInput(new Synapse(previousLayerNeurons.get(i), weights[i]));
                }
            }

        }
    }

    public void feedForward() {

        int biasCount = hasBias() ? 1 : 0;

        for(int i = biasCount; i < neurons.size(); i++) {
            neurons.get(i).activate();
        }
    }

    public Layer getPreviousLayer() {
        return previousLayer;
    }

    void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

    public Layer getNextLayer() {
        return nextLayer;
    }

    void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public boolean isOutputLayer() {
        return nextLayer == null;
    }

    public boolean hasBias() {
        return bias != null;
    }
}
