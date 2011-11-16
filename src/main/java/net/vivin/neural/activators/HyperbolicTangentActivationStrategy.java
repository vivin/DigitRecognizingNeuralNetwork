package net.vivin.neural.activators;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 3:07 PM
 */
public class HyperbolicTangentActivationStrategy implements ActivationStrategy, Serializable {
    public double activate(double weightedSum) {
        double a = Math.exp(weightedSum);
        double b = Math.exp(-weightedSum);
        return ((a-b)/(a+b));
    }

    public double derivative(double weightedSum) {
        return 1 - Math.pow(activate(weightedSum), 2.0);
    }

    public HyperbolicTangentActivationStrategy copy() {
        return new HyperbolicTangentActivationStrategy();
    }
}
