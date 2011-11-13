package net.vivin.neural.generator;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/11/11
 * Time: 6:06 PM
 */
public class TrainingData {

    private double[][] inputs;
    private double[][] outputs;

    public TrainingData(double[][] inputs, double[][] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    public double[][] getInputs() {
        return inputs;
    }

    public double[][] getOutputs() {
        return outputs;
    }
}
