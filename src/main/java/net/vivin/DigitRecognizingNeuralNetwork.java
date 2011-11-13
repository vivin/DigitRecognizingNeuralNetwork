package net.vivin;

import net.vivin.digit.DigitImage;
import net.vivin.neural.Backpropagator;
import net.vivin.neural.Layer;
import net.vivin.neural.NeuralNetwork;
import net.vivin.neural.Neuron;
import net.vivin.neural.activators.LinearActivationStrategy;
import net.vivin.neural.activators.SigmoidActivationStrategy;
import net.vivin.service.DigitImageLoadingService;

import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/11/11
 * Time: 10:02 AM
 */
public class DigitRecognizingNeuralNetwork {

    public static void main(String[] args) throws IOException {
        DigitImageLoadingService service = new DigitImageLoadingService("/train/train-labels-idx1-ubyte.dat", "/train/train-images-idx3-ubyte.dat");

        List<DigitImage> images = service.loadDigitImages();

        NeuralNetwork neuralNetwork = new NeuralNetwork();

        Neuron inputBias = new Neuron(new LinearActivationStrategy());
        inputBias.setOutput(1);

        Layer inputLayer = new Layer(null, inputBias);

        for(int i = 0; i < DigitImageLoadingService.ROWS * DigitImageLoadingService.COLUMNS; i++) {
            Neuron neuron = new Neuron(new SigmoidActivationStrategy());
            neuron.setOutput(0);
            inputLayer.addNeuron(neuron);
        }

        Neuron hiddenBias = new Neuron(new LinearActivationStrategy());
        hiddenBias.setOutput(1);

        Layer hiddenLayer = new Layer(inputLayer, hiddenBias);

        long numberOfHiddenNeurons = Math.round((2.0 / 3.0) * (DigitImageLoadingService.ROWS * DigitImageLoadingService.COLUMNS) + 10);

        for(int i = 0; i < numberOfHiddenNeurons; i++) {
            Neuron neuron = new Neuron(new SigmoidActivationStrategy());
            neuron.setOutput(0);
            hiddenLayer.addNeuron(neuron);
        }

        Layer outputLayer = new Layer(hiddenLayer);

        //10 output neurons - 1 for each digit
        for(int i = 0; i < 10; i++) {
            Neuron neuron = new Neuron(new SigmoidActivationStrategy());
            neuron.setOutput(0);
            outputLayer.addNeuron(neuron);
        }

        neuralNetwork.addLayer(inputLayer);
        neuralNetwork.addLayer(hiddenLayer);
        neuralNetwork.addLayer(outputLayer);

        //Build a map using the image label as a key. The value is a list of images. This will help us pick random images
        Map<Integer, List<DigitImage>> labelToImageListMap = new HashMap<Integer, List<DigitImage>>();

        for(DigitImage image : images) {

            if(labelToImageListMap.get(image.getLabel()) == null) {
                labelToImageListMap.put(image.getLabel(), new ArrayList<DigitImage>());
            }

            labelToImageListMap.get(image.getLabel()).add(image);
        }

        int[] digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

        double error = 0;
        double[][] inputs = new double[10][DigitImageLoadingService.ROWS * DigitImageLoadingService.COLUMNS];
        double[][] outputs = new double[10][10];

        Backpropagator backpropagator = new Backpropagator(neuralNetwork, 0.1, 0.9);

             //initialize inputs and outputs
           /* digits = shuffle(digits);
            for(int i = 0; i < 10; i++) {
                DigitImage randomImage = getRandomImage(digits[i], labelToImageListMap);
                inputs[i] = randomImage.getData();
                outputs[i] = getOutputFor(digits[i]);
            }

        for(int i = 0; i < digits.length; i++) {
            System.out.print(digits[i] + ", ");
        }*/

        System.out.println();

        do {

            //initialize inputs and outputs
            digits = shuffle(digits);
            for(int i = 0; i < 10; i++) {
                inputs[i] = getRandomImage(digits[i], labelToImageListMap).getData();
                outputs[i] = getOutputFor(digits[i]);
            }

            error = backpropagator.backpropagate(inputs, outputs);
            System.out.println(error);

            int digit = new Random().nextInt(10);

            DigitImage randomImage = getRandomImage(new Random().nextInt(10), labelToImageListMap);
            neuralNetwork.setInputs(randomImage.getData());
            double[] output = neuralNetwork.getOutput();

            int recognized = 0;
            double best = output[0];
            for(int i = 0; i < output.length; i++) {
                if(output[i] > best) {
                    best = output[i];
                    recognized = i;
                }
            }

            System.out.println("Recognized image of " + digit + " as a " + recognized + " with output equal to " + best + ": expected " + explode(getOutputFor(digit)) + " got " + explode(output) + "\n");


        } while(error > 0.00001);


        System.out.println(images.size());
    }

    private static String explode(double[] array) {
        String string = "[";

        for (double number : array) {
            string += number + ", ";
        }

        Pattern pattern = Pattern.compile(", $", Pattern.DOTALL);
        Matcher matcher = pattern.matcher(string);
        string = matcher.replaceAll("");

        return string + "]";
    }

    private static DigitImage getRandomImage(int label, Map<Integer, List<DigitImage>> map) {

        Random random = new Random();
        List<DigitImage> images = map.get(label);

        return images.get(random.nextInt(images.size()));
    }

    private static double[] getOutputFor(int label) {
        double[] output = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        output[label] = 1;
        return output;
    }

    private static int[] shuffle(int[] array) {

        Random random = new Random();
        for(int i = array.length - 1; i > 0; i--) {

            int index = random.nextInt(i + 1);
            int temp = array[i];

            array[i] = array[index];
            array[index] = temp;
        }

        return array;
    }
}
