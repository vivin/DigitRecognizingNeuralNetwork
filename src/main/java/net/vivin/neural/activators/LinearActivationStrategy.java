package net.vivin.neural.activators;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/11/11
 * Time: 12:11 PM
 */
public class LinearActivationStrategy implements ActivationStrategy {
    public double activate(double weightedSum) {
        return weightedSum;
    }

    public double derivative(double weightedSum) {
        return 0;
    }

    public ActivationStrategy copy() {
        return new LinearActivationStrategy();
    }
}
