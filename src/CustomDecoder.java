/**
 * @author: ccann
 */

import weka.classifiers.lazy.IBk;

import java.util.ArrayList;
import java.util.LinkedList;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public class CustomDecoder {

    private static String root = "";
    private static String fp = "file:"+root+"resources/green.wav";

    private static int NUM_BINS = 5;


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
    private static double[] getMetaFeatures(ArrayList<Float> features){

        SummaryStatistics s = new SummaryStatistics();
        for (Float feature : features) {
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


    public static void main(String args[]) {
        Utterance utt = new Utterance(fp);
        ArrayList<ArrayList<Float>> bins = getBinnedFeatures(utt);
        ArrayList<double[]> metas = new ArrayList<double[]>();

        for (ArrayList<Float> bin : bins){
            metas.add(getMetaFeatures(bin));
        }

        System.out.println(metas);




        //initSphinxFrontEnd(fp);
        //ArrayList<float[]> features = getFeatures();
        //printFeatures(features);
    }





}
