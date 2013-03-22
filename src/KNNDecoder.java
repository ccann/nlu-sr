/**
 * @author: ccann
 *
 * This program is a fairly simple speech recognizer built for
 * CS 150-06: Situated Natural Language Understanding on Robots. It reads speech from a file (.wav) and
 * outputs the recognized utterance. Right now the dictionary of words includes three color words:
 * red, green and white.
 *
 * The decoder operates as follows:
 *    divide the utterance (e.g. "red") into NUM_BINS bins. Each bin represents a portion of the utterance
 *                                               "red" ==>       [ bin1 -- bin2 -- bin3 -- bin4 -- bin5 ]
 *    Each bin contains acoustic features derived from a portion of the input utterance. These features are currently
 *    composed of cepstra, which are derived from the Sphinx4 frontend:
 *    http://cmusphinx.sourceforge.net/sphinx4/javadoc/edu/cmu/sphinx/frontend/feature/DeltasFeatureExtractor.html
 *
 *    A k-nearest-neighbors classifier (WEKA's IBk) is trained on each individual bin, i.e. on the features of each
 *    portion (1/5 if NUM_BINS == 5) of each utterance in the training set. During testing each classifier classifies
 *    a portion of the test utterance as one of the words in the dictionary (e.g. red, green, or white). The
 *    cumulative classification tally for each dictionary word is retained in the scores list. The KNNDecoder
 *    determines the most likely classification of the whole utterance based on the dictionary word with the highest
 *    score (i.e. highest number of respective bin classifications) relative to the others.
 *
 */

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import weka.core.*;

public class KNNDecoder {

    private static String pathToTrainingFiles = "resources/";
    private static String pathToTestingFiles = "test/";

    private static int NUM_BINS = 5;
    private static int NUM_FEATURES = 3;
    private static int TRAINING_SET_SIZE = 127;
    private static Instances[] trainingSets;
    private static Instance testInstance;
    private static ArrayList<Attribute> fv;
    private static int k = 7;  // should be odd
    private static ArrayList<Classifier> binClassifiers = new ArrayList<Classifier>(NUM_BINS);
    private static String[] dictionary = {"red","green","white"};
    private static ArrayList<Integer> score = new ArrayList<Integer>(dictionary.length);

    /**
     * dumps the features from the feature vectors into NUM_BINS bins
     * @param utt  the utterance from which we're getting the features
     * @return a list containing the the feature bins
     */
    private static ArrayList<ArrayList<Float>> getBinnedFeatures (Utterance utt){

        ArrayList<ArrayList<Float>> bins = new ArrayList<ArrayList<Float>>(NUM_BINS);
        LinkedList<float[]> featureVectors = utt.getFeatures();

        // the number of feature vectors to be dumped into each bin
        int binSize = utt.getNumWindows() / NUM_BINS;

        for (int i = 0; i < NUM_BINS; i++){

            ArrayList<Float> bin = new ArrayList<Float>(binSize);
            for(int j =0; j < binSize ; j++){

                // pop first feature vector in featureVectors
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
     * Derive a meta feature vector for each bin. This implementation is pretty naive as it
     * uses max,min,mean,stdev, and variance across a collection of cepstra
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

        //double[] metaFeatures = {max, min, mean, stdev, variance};
        double [] metaFeatures = {max,min,mean};

        return metaFeatures;
    }

    //private static void initClassifier()

    /**
     * initialize the training set attributes
     */
    private static void initTrainingSets(){

        Attribute max = new Attribute("max");
        Attribute min = new Attribute("min");
        Attribute mean = new Attribute("mean");
        Attribute stdev = new Attribute("stdev");
        Attribute variance = new Attribute("variance");

        ArrayList<String> classes = new ArrayList<String>(dictionary.length);
        for(int i=0; i<dictionary.length;i++){
            classes.add(dictionary[i]);
        }
        Attribute classAttribute = new Attribute("word", classes);

        fv = new ArrayList<Attribute>(NUM_FEATURES+1);
        fv.add(max);
        fv.add(min);
        fv.add(mean);
        //fv.add(stdev);
        //fv.add(variance);
        fv.add(classAttribute);

        for (int i =0; i < trainingSets.length; i++){
            trainingSets[i] = new Instances("Training",fv, TRAINING_SET_SIZE);
            trainingSets[i].setClassIndex(NUM_FEATURES);
        }

    }

    /**
     * add the meta features for this utterance to the training or test set
     * @param utt  utterance from which the meta features came
     * @param metaFeatures  meta features to add to training set
     */
    private static Instance createInstance(Utterance utt, double[] metaFeatures){

        DenseInstance in = new DenseInstance(NUM_FEATURES+1);
        in.setValue(fv.get(0), metaFeatures[0]);
        in.setValue(fv.get(1), metaFeatures[1]);
        in.setValue(fv.get(2), metaFeatures[2]);
      //  in.setValue(fv.get(3), metaFeatures[3]);
      //  in.setValue(fv.get(4), metaFeatures[4]);

        String filename = utt.getFilePath();
        for (String word : dictionary){
            if (filename.contains(word)){
                in.setValue(fv.get(NUM_FEATURES),word);
                break;
            }
        }
        return in;
    }

    /**
     * build the KNN classifier for each bin
     */
    private static void trainClassifiers(){
        File dir = new File(pathToTrainingFiles);
        File[] files = dir.listFiles();

        for (File file : files){
            if(file.isFile() && (!file.getName().contains("DS"))){
                Utterance utt = new Utterance(pathToTrainingFiles + file.getName());

                ArrayList<ArrayList<Float>> bins = getBinnedFeatures(utt);
                for(int i=0; i < NUM_BINS; i++){
                    // add meta features of bin to respective training set
                    Instance newInst = createInstance(utt, getMetaFeatures(bins.get(i)));
                    trainingSets[i].add(newInst);
                }
            }
        }

        for (Instances trainingSet : trainingSets) {
            // create a classifier for each bin
            try{
                Classifier knn = new IBk(k);
                knn.buildClassifier(trainingSet);
                binClassifiers.add(knn);
            }
            catch(Exception ex){
                System.out.println("ERROR: classifier NOT generated successfully");
            }
        }
    }

    /**
     * test an utterance against the training set
     * @param utt  the utterance to be tested
     */
    private static void test(Utterance utt){

        ArrayList<ArrayList<Float>> bins = getBinnedFeatures(utt);
        for(int i = 0; i<NUM_BINS; i++){
            testInstance = createInstance(utt, getMetaFeatures(bins.get(i)));
            // specify test instance belongs to the training set so that it can inherit from that set description
            testInstance.setDataset(trainingSets[0]);
            double[] dist;
            try{
                dist = binClassifiers.get(i).distributionForInstance(testInstance);
                System.out.println("red: " + dist[0] +
                                   "  green: " + dist[1] +
                                   "  white: " + dist[2]);
                updateScore(dist);
            }
            catch(Exception ex){
                System.out.println("Problem with testing test instance");
            }
        }
    }

    /**
     * update the score of each word in the dictionary
     * @param dist  the word probability distribution
     */
    private static void updateScore(double[] dist){
        ArrayList<Double> d = new ArrayList<Double>();
        for(int i = 0; i < dist.length;i++){
            d.add(dist[i]);
        }
        int scoreIndex = d.indexOf(Collections.max(d));
        score.set(scoreIndex, score.get(scoreIndex) + 1);
    }

    public static void main(String args[]) {
        // TRAINING
        trainingSets = new Instances[NUM_BINS];
        initTrainingSets();
        trainClassifiers();


        File dir = new File(pathToTestingFiles);
        File[] files = dir.listFiles();
        for (File file : files){
            if(file.isFile() && (!file.getName().contains("DS"))){

                // TESTING
                Utterance testUtt = new Utterance(pathToTestingFiles+file.getName());
                System.out.println("\nTesting: " + file.getName());
                for (int i = 0; i < dictionary.length;i++){
                    score.add(0);
                }
                test(testUtt);

                // SCORING and REPORTING
                double highestScore = Collections.max(score);
                System.out.println(dictionary[score.indexOf(Collections.max(score))] + " " +
                        (int)highestScore + "/5" );

                score = new ArrayList<Integer>(dictionary.length);

            }
        }
    }
}

