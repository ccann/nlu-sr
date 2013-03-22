/**
 * @author: ccann
 */

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedList;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import weka.core.*;

public class CustomDecoder {

    private static String fp = "file:resources/";

    private static int NUM_BINS = 5;
    // number of features includes the "class" feature
    private static int NUM_FEATURES = 6;
    private static int TRAINING_SET_SIZE = 45;
    private static Instances trainingSet;
    private static ArrayList<Attribute> fv;
    private static int k = 3;  // should be odd
    private static ArrayList<Classifier> binClassifiers;
    private static String[] dictionary = {"red","green","white"};

    /**
     * print the features
     * @param features  the features to print
     */
    private static void printFeatures(ArrayList<float[]> features) {
        for(int i=0;i<features.size();i++){
            for(int j=0;j<features.get(i).length;j++){
                System.out.print(features.get(i)[j] + " ");
            }
            System.out.println();
        }
    }

    /**
     * dumps the cepstra from the cepstra vectors into NUM_BINS bins
     * @param utt  the utterance from which we're getting the cepstra
     * @return a list containing the the cepstra bins
     */
    private static ArrayList<ArrayList<Float>> getBinnedFeatures (Utterance utt){

        ArrayList<ArrayList<Float>> bins = new ArrayList<ArrayList<Float>>(NUM_BINS);
        LinkedList<float[]> featureVectors = utt.getFeatures();

        // the number of feature vectors to be dumped into each bin
        int binSize = utt.getNumWindows() / NUM_BINS;

        for (int i = 0; i < NUM_BINS; i++){

            ArrayList<Float> bin = new ArrayList<Float>(binSize);
            for(int j =0; j < binSize ; j++){

                // pop first feature vector in cepstraVectors
                float[] vec = featureVectors.remove();
                for (float feature : vec){
                    // add the floats from the vector to the bin
                    bin.add(feature);
                }
            }
            bins.add(bin);
        }

        // in the case that there are extra feature vectors (remainder of binSize), add the feature to the last bin.
        if (! featureVectors.isEmpty()){
            for (float[] cepstra : featureVectors){
                for (int i=0;i<cepstra.length;i++){
                    bins.get(NUM_BINS-1).add(cepstra[i]);
                }
            }
        }
        return bins;
    }

    /**
     * Derive a meta feature vector for each bin
     *
     */
    private static double[] getMetaFeatures(ArrayList<Float> bin){

        SummaryStatistics s = new SummaryStatistics();
        for (Float feature : bin) {
            s.addValue(feature.doubleValue());
        }

        double max = s.getMax();
        double min = s.getMin();
        double mean = s.getMean();
        double stdev = s.getStandardDeviation();
        double variance = s.getVariance();

        double[] metaFeatures = {max, min, mean, stdev, variance};
        return metaFeatures;
    }

    //private static void initClassifier()

    /**
     * initialize the training set attributes
     */
    private static void initTrainingSet(){

        Attribute max = new Attribute("max");
        Attribute min = new Attribute("min");
        Attribute mean = new Attribute("mean");
        Attribute stdev = new Attribute("stdev");
        Attribute variance = new Attribute("variance");

        ArrayList<String> classes = new ArrayList<String>(dictionary.length);
        Attribute classAttribute = new Attribute("word", classes);

        fv = new ArrayList<Attribute>(NUM_FEATURES);
        fv.add(max);
        fv.add(min);
        fv.add(mean);
        fv.add(stdev);
        fv.add(variance);
        fv.add(classAttribute);

        trainingSet = new Instances("Training",fv, TRAINING_SET_SIZE);
        trainingSet.setClassIndex(NUM_FEATURES);
    }

    /**
     * add the meta features for this utterance to the training set
     * @param utt  utterance from which the meta features came
     * @param metaFeatures  meta features to add to training set
     */
    private static void addToTrainingSet(Utterance utt, double[] metaFeatures){

        DenseInstance in = new DenseInstance(NUM_FEATURES);
        in.setValue(fv.get(0), metaFeatures[0]);
        in.setValue(fv.get(1), metaFeatures[1]);
        in.setValue(fv.get(2), metaFeatures[2]);
        in.setValue(fv.get(3), metaFeatures[3]);
        in.setValue(fv.get(4), metaFeatures[4]);

        String filename = utt.getFilePath();
        for (String word : dictionary){
            if (filename.contains(word)){
                in.setValue(fv.get(5),word);
                break;
            }
        }
        trainingSet.add(in);
    }

    /**
     * build the KNN classifier for each bin
     */
    private static void trainClassifiers(){
        File dir = new File(fp);
        File[] files = dir.listFiles();

        for (File file : files){
            if(file.isFile()){
                Utterance utt = new Utterance(file.getName());
                for(ArrayList<Float> bin : getBinnedFeatures(utt)){

                    // create a classifier for each bin
                    addToTrainingSet(utt, getMetaFeatures(bin));
                    try{
                        Classifier knn = new IBk(k);
                        knn.buildClassifier(trainingSet);

                        binClassifiers.add(knn);
                    }
                    catch(Exception ex){
                        System.out.println("ERROR: classifier not being built");
                    }
                }
            }
        }
    }

    public static void main(String args[]) {

        initTrainingSet();
        trainClassifiers();
        // @TODO: create scorer: tally each bin classification
        // @TODO: pipeline for testing, might need some new audio files

    }





}
