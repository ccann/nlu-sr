/**
 * @author: ccann
 */

import edu.cmu.sphinx.frontend.*;
import edu.cmu.sphinx.frontend.feature.DeltasFeatureExtractor;
import edu.cmu.sphinx.frontend.feature.LiveCMN;
import edu.cmu.sphinx.frontend.filter.Preemphasizer;
import edu.cmu.sphinx.frontend.frequencywarp.MelFrequencyFilterBank;
import edu.cmu.sphinx.frontend.transform.DiscreteCosineTransform;
import edu.cmu.sphinx.frontend.transform.DiscreteFourierTransform;
import edu.cmu.sphinx.frontend.util.AudioFileDataSource;
import edu.cmu.sphinx.frontend.window.RaisedCosineWindower;

import weka.classifiers.lazy.IBk;

import java.net.URL;
import java.util.ArrayList;

public class CustomDecoder {

    private static FrontEnd frontend;
    private static String root = "/home/cody/IdeaProjects/nlu-speech-recognizer/";
    private static String fp = "file:"+root+"resources/green.wav";

    private static int NUM_CEPSTRA = 13;

    /**
     * initialize the Sphinx4 frontend pipeline. This pipeline does the DSP for the audio inputs.
     */
    protected static void initSphinxFrontEnd(String fp) {
        AudioFileDataSource audioDataSource = new AudioFileDataSource(3200, null);

        try {
            URL audioURL = new URL(fp);
            audioDataSource.setAudioFile(audioURL, null);
        }
        catch (Exception ex)
        {
            System.out.println("file not found: " + fp);
        }

        Preemphasizer preemphasizer = new Preemphasizer(
                0.97 // preemphasisFactor
        );

        RaisedCosineWindower windower = new RaisedCosineWindower(
                0.46, // double alpha
                25.625f, // windowSizeInMs
                10.0f // windowShiftInMs
        );

        DiscreteFourierTransform fft = new DiscreteFourierTransform(
                -1, // numberFftPoints
                false // invert
        );

        MelFrequencyFilterBank melFilterBank = new MelFrequencyFilterBank(
                130.0, // minFreq,
                6800.0, // maxFreq,
                40 // numberFilters
        );

        DiscreteCosineTransform dct = new DiscreteCosineTransform(
                40, // numberMelFilters,
                13  // cepstrumSize
        );

        LiveCMN cmn = new LiveCMN(
                12.0, // initialMean,
                100,  // cmnWindow,
                160   // cmnShiftWindow
        );

        DeltasFeatureExtractor featureExtraction = new DeltasFeatureExtractor(
                3 // window
        );

        ArrayList pipeline = new ArrayList();
        pipeline.add(audioDataSource);
        pipeline.add(preemphasizer);
        pipeline.add(windower);
        pipeline.add(fft);
        pipeline.add(melFilterBank);
        pipeline.add(dct);
        pipeline.add(cmn);
        pipeline.add(featureExtraction);

        frontend = new FrontEnd(pipeline);
    }

    /**
     * get NUM_CEPSTRA cepstra from the sphinx4 frontend.
     * @return cepstra for each window
     */
    private static ArrayList<float[]> getCepstra() {
        ArrayList<float[]> floatArrs = new ArrayList<float[]>();
        Data data = frontend.getData();


        while (! (data instanceof DataEndSignal)){
            if (! (data instanceof DataStartSignal))
            {
                // cast data to FloatData and retrieve the float array, add it to floatArrs
                floatArrs.add(((FloatData) data).getValues());
            }
            data = frontend.getData();
        }

        ArrayList<float[]> cepstra = new ArrayList<float[]>(floatArrs.size());

        for(int i=0;i<floatArrs.size();i++)
        {
            float[] temp = new float[NUM_CEPSTRA];
            for (int j=0;j<NUM_CEPSTRA;j++){
                temp[j] = floatArrs.get(i)[j];
            }
            cepstra.add(temp);
        }
        return cepstra;
    }


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

    public static void main(String args[]) {
        initSphinxFrontEnd(fp);
        ArrayList<float[]> features = getCepstra();
        printFeatures(features);
    }





}
