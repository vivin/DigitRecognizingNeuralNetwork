package net.vivin.neural;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 12:03 PM
 */
public class Synapse implements Serializable {

    private Neuron sourceNeuron;
    private double weight;

    public Synapse(Neuron sourceNeuron, double weight) {
        this.sourceNeuron = sourceNeuron;
        this.weight = weight;
    }

    public Neuron getSourceNeuron() {
        return sourceNeuron;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }
}
