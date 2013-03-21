/**
 * @author: ccann
 */

import weka.classifiers.lazy.IBk;

import java.util.ArrayList;
import java.util.LinkedList;

public class CustomDecoder {

    private static String root = "/home/cody/IdeaProjects/nlu-speech-recognizer/";
    private static String fp = "file:"+root+"resources/green.wav";

    private static int NUM_BINS = 5;

//    private static ArrayList<Integer> compressCepstra(ArrayList<float[]> cepstra){
//
//    }

    /**
     * print the FloatData features (cepstra, in this case)
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
    private ArrayList<ArrayList<Float>> getBinnedCepstra (Utterance utt){

        ArrayList<ArrayList<Float>> bins = new ArrayList<ArrayList<Float>>(NUM_BINS);
        LinkedList<float[]> cepstraVectors = utt.getCepstra();

        // the number of cepstra vectors to be dumped into each bin
        int binSize = utt.getNumWindows() / NUM_BINS;

        for (int i = 0; i < NUM_BINS; i++){

            ArrayList<Float> bin = new ArrayList<Float>(binSize);
            for(int j =0; j < binSize ; j++){

                // pop first cepstra vector in cepstraVectors
                float[] vec = cepstraVectors.remove();
                for (float cepstrum : vec){
                    // add the floats from the vector to the bin
                    bin.add(cepstrum);
                }
            }
            bins.add(bin);
        }

        // in the case that there are extra cepstra vectors (remainder of binSize), add the cepstra to the last bin.
        if (! cepstraVectors.isEmpty()){
            for (float[] cepstra : cepstraVectors){
                for (int i=0;i<cepstra.length;i++){
                    bins.get(NUM_BINS-1).add(cepstra[i]);
                }
            }
        }
        return bins;
    }

    public static void main(String args[]) {
        //initSphinxFrontEnd(fp);
        //ArrayList<float[]> features = getCepstra();
        //printFeatures(features);
    }





}
