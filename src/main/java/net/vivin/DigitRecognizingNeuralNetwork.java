package net.vivin;

import net.vivin.digit.DigitImage;
import net.vivin.digit.generator.DigitTrainingDataGenerator;
import net.vivin.neural.Backpropagator;
import net.vivin.neural.Layer;
import net.vivin.neural.NeuralNetwork;
import net.vivin.neural.Neuron;
import net.vivin.neural.activators.LinearActivationStrategy;
import net.vivin.neural.activators.SigmoidActivationStrategy;
import net.vivin.neural.generator.TrainingData;
import net.vivin.service.DigitImageLoadingService;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
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

        DigitImageLoadingService trainingService = new DigitImageLoadingService("/train/train-labels-idx1-ubyte.dat", "/train/train-images-idx3-ubyte.dat");
        DigitImageLoadingService testService = new DigitImageLoadingService("/test/t10k-labels-idx1-ubyte.dat", "/test/t10k-images-idx3-ubyte.dat");

        NeuralNetwork neuralNetwork = new NeuralNetwork("Digit Recognizing Neural Network");

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

        DigitTrainingDataGenerator trainingDataGenerator = new DigitTrainingDataGenerator(trainingService.loadDigitImages());
        Backpropagator backpropagator = new Backpropagator(neuralNetwork, 0.1, 0.9);
        backpropagator.train(trainingDataGenerator, 0.005);
        neuralNetwork.persist();

        DigitTrainingDataGenerator testDataGenerator = new DigitTrainingDataGenerator(testService.loadDigitImages());
        TrainingData testData = testDataGenerator.getTrainingData();

        for(int i = 0; i < testData.getInputs().length; i++) {
            double[] input = testData.getInputs()[i];
            double[] output = testData.getOutputs()[i];

            int digit = 0;
            boolean found = false;
            while(digit < 10 && !found) {
                found = (output[digit] == 1);
                digit++;
            }

            neuralNetwork.setInputs(input);
            double[] receivedOutput = neuralNetwork.getOutput();

            double max = receivedOutput[0];
            double recognizedDigit = 0;
            for(int j = 0; j < receivedOutput.length; j++) {
                if(receivedOutput[j] > max) {
                    max = receivedOutput[j];
                    recognizedDigit = j;
                }
            }

            System.out.println("Recognized " + (digit - 1) + " as " + recognizedDigit + ". Corresponding output value was " + max);
        }
    }
}
