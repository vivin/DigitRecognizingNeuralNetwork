package net.vivin.digit;

import net.vivin.neural.activators.SigmoidActivationStrategy;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/11/11
 * Time: 10:05 AM
 */
public class DigitImage {

    private int label;
    private double[] data;

    public DigitImage(int label, byte[] data) {
        this.label = label;

        this.data = new double[data.length];

        for(int i = 0; i < this.data.length; i++) {
            this.data[i] =  data[i] & 0xFF; //convert to unsigned
        }

        otsu();
    }

    //Uses Otsu's Threshold algorithm to convert from grayscale to black and white
    private void otsu() {
        int[] histogram = new int[256];

        for(double datum : data) {
            histogram[(int) datum]++;
        }

        double sum = 0;
        for(int i = 0; i < histogram.length; i++) {
            sum += i * histogram[i];
        }

        double sumB = 0;
        int wB = 0;
        int wF = 0;

        double maxVariance = 0;
        int threshold = 0;

        int i = 0;
        boolean found = false;

        while(i < histogram.length && !found) {
            wB += histogram[i];

            if(wB != 0) {
                wF = data.length - wB;

                if(wF != 0) {
                    sumB += (i * histogram[i]);

                    double mB = sumB / wB;
                    double mF = (sum - sumB) / wF;

                    double varianceBetween = wB * Math.pow((mB - mF), 2);

                    if(varianceBetween > maxVariance) {
                        maxVariance = varianceBetween;
                        threshold = i;
                    }
                }

                else {
                    found = true;
                }
            }

            i++;
        }

/*        System.out.println(label + ": threshold is " + threshold);

        for(i = 0; i < data.length; i++) {
            if(i % 28 == 0) {
                System.out.println("<br />");
            }

            System.out.print("<span style='color:rgb(" + (int) (255 - data[i]) + ", " + (int) (255 - data[i]) + ", " + (int) (255 - data[i]) + ")'>&#9608;</span>");
        } */

        for(i = 0; i < data.length; i++) {
            data[i] = data[i] <= threshold ? 0 : 1;
        }
/*
        if(label == 7 || label == 9) {
            for(i = 0; i < data.length; i++) {
                if(i % 28 == 0) {
                    System.out.println("");
                }

                System.out.print(data[i] == 1 ? "#" : " ");
            }
        }*/
    }

    public int getLabel() {
        return label;
    }

    public double[] getData() {
        return data;
    }
}
