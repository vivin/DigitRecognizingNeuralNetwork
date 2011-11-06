package net.vivin.neural;

import net.vivin.neural.activators.ActivationStrategy;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 11:52 AM
 */
public class Neuron {

    private List<Synapse> inputs;
    private ActivationStrategy activationStrategy;
    private double output;

    public Neuron(ActivationStrategy activationStrategy) {
        inputs = new ArrayList<Synapse>();
        this.activationStrategy = activationStrategy;
    }

    public void addInput(Synapse input) {
        inputs.add(input);
    }

    public List<Synapse> getInputs() {
        return this.inputs;
    }

    public double getOutput() {
        return this.output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public ActivationStrategy getActivationStrategy() {
        return activationStrategy;
    }
}
