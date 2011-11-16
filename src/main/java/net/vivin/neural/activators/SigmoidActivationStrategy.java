package net.vivin.neural.activators;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 6:45 PM
 */
public class SigmoidActivationStrategy implements ActivationStrategy, Serializable {
    public double activate(double weightedSum) {
        return 1.0 / (1 + Math.exp(-1.0 * weightedSum));
    }

    public double derivative(double weightedSum) {
        return weightedSum * (1.0 - weightedSum);
    }

    public SigmoidActivationStrategy copy() {
        return new SigmoidActivationStrategy();
    }
}
