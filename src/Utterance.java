import edu.cmu.sphinx.frontend.*;
import edu.cmu.sphinx.frontend.feature.DeltasFeatureExtractor;
import edu.cmu.sphinx.frontend.feature.LiveCMN;
import edu.cmu.sphinx.frontend.filter.Preemphasizer;
import edu.cmu.sphinx.frontend.frequencywarp.MelFrequencyFilterBank;
import edu.cmu.sphinx.frontend.transform.DiscreteCosineTransform;
import edu.cmu.sphinx.frontend.transform.DiscreteFourierTransform;
import edu.cmu.sphinx.frontend.util.AudioFileDataSource;
import edu.cmu.sphinx.frontend.window.RaisedCosineWindower;

import java.net.URL;
import java.util.ArrayList;
import java.util.LinkedList;

/**
 * @author: ccann
 */
public class Utterance {

    private int numWindows;
    private String filepath;
    private FrontEnd frontend;
    private static int NUM_FEATURES = 13;


    public void setNumWindows(int n){
        numWindows = n;
    }

    public int getNumWindows(){
        return numWindows;
    }

    public String getFilePath(){
        return filepath;
    }

    /**
     * get the feature arrays from the sphinx4 frontend.
     * @return features for each window
     */
    public LinkedList<float[]> getFeatures() {
        initFrontend(filepath);

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

        this.numWindows = floatArrs.size();

        LinkedList<float[]> features = new LinkedList<float[]>();

        for(int i=0;i<floatArrs.size();i++)
        {
            float[] temp = new float[NUM_FEATURES];
            for (int j=0;j< NUM_FEATURES;j++){
                temp[j] = floatArrs.get(i)[j];
            }
            features.add(temp);
        }
        return features;
    }

    /**
     * initialize the Sphinx4 frontend pipeline. This pipeline does the DSP for the audio inputs.
     */
    protected void initFrontend(String fp) {
        AudioFileDataSource audioDataSource = new AudioFileDataSource(3200, null);

        try {
            URL audioURL = new URL("file:"+fp);
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

    public Utterance(String f){
        this.filepath = f;
        initFrontend(this.filepath);
    }

}
