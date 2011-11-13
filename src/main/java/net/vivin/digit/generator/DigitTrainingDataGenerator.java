package net.vivin.digit.generator;

import net.vivin.digit.DigitImage;
import net.vivin.neural.generator.TrainingData;
import net.vivin.neural.generator.TrainingDataGenerator;
import net.vivin.service.DigitImageLoadingService;

import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/11/11
 * Time: 6:09 PM
 */
public class DigitTrainingDataGenerator implements TrainingDataGenerator {

    Map<Integer, List<DigitImage>> labelToDigitImageListMap;
    int[] digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    public DigitTrainingDataGenerator(List<DigitImage> digitImages) {
        labelToDigitImageListMap = new HashMap<Integer, List<DigitImage>>();

        for (DigitImage digitImage: digitImages) {

            if (labelToDigitImageListMap.get(digitImage.getLabel()) == null) {
                labelToDigitImageListMap.put(digitImage.getLabel(), new ArrayList<DigitImage>());
            }

            labelToDigitImageListMap.get(digitImage.getLabel()).add(digitImage);
        }
    }

    public TrainingData getTrainingData() {
        digits = shuffle(digits);

        double[][] inputs = new double[10][DigitImageLoadingService.ROWS * DigitImageLoadingService.COLUMNS];
        double[][] outputs = new double[10][10];

        for(int i = 0; i < 10; i++) {
            inputs[i] = getRandomImageForLabel(digits[i]).getData();
            outputs[i] = getOutputFor(digits[i]);
        }

        return new TrainingData(inputs, outputs);
    }

    private int[] shuffle(int[] array) {

        Random random = new Random();
        for(int i = array.length - 1; i > 0; i--) {

            int index = random.nextInt(i + 1);

            int temp = array[i];
            array[i] = array[index];
            array[index] = temp;
        }

        return array;
    }

    private DigitImage getRandomImageForLabel(int label) {
        Random random = new Random();
        List<DigitImage> images = labelToDigitImageListMap.get(label);
        return images.get(random.nextInt(images.size()));
    }

    private double[] getOutputFor(int label) {
        double[] output = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        output[label] = 1;
        return output;
    }
}
