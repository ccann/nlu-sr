/**
 * Created with IntelliJ IDEA.
 * User: cody
 * Date: 3/19/13
 * Time: 5:05 PM
 */

import edu.cmu.sphinx.frontend.FrontEnd;
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


public class CustomDecoder {

    private FrontEnd frontend;
    private String root = "/home/cody/IdeaProjects/nlu-speech-recognizer";


    /**
     * initialize the Sphinx4 frontend pipeline. This pipeline does the DSP for the audio inputs.
     */
    protected void initFrontEnd() {
        AudioFileDataSource audioDataSource = new AudioFileDataSource(3200, null);
        String filepath = "file:"+root+"resources/red.wav";

        try {
            URL audioURL = new URL(filepath);
            audioDataSource.setAudioFile(audioURL, null);
        }
        catch (Exception ex)
        {
            System.out.println("file not found: " + filepath);
        }

        Preemphasizer premphasizer = new Preemphasizer(
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
        pipeline.add(premphasizer);
        pipeline.add(windower);
        pipeline.add(fft);
        pipeline.add(melFilterBank);
        pipeline.add(dct);
        pipeline.add(cmn);
        pipeline.add(featureExtraction);

        this.frontend = new FrontEnd(pipeline);
    }


}
