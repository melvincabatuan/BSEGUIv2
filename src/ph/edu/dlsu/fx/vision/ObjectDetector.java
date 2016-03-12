package ph.edu.dlsu.fx.vision;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.Rect;
import org.opencv.core.MatOfRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

/**
 * Created by cobalt on 3/7/16.
 */
public class ObjectDetector {

    private static final String cascadePath = "res/haarcascades/haarcascade_frontalface_alt.xml";
    private CascadeClassifier classifier;
    private int absoluteFaceSize;
    private Rect objectRoi = null;

    public ObjectDetector() {
        this.classifier = new CascadeClassifier();
        this.classifier.load(cascadePath);
        this.absoluteFaceSize = 0;
    }

    public void detectAndDisplay(Mat frame) {
        MatOfRect faces = new MatOfRect();
        Mat grayFrame = new Mat();

        // convert the frame in gray scale
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        // equalize the frame histogram to improve the result
        Imgproc.equalizeHist(grayFrame, grayFrame);

        // compute minimum face size (20% of the frame height, in our case)
        if (this.absoluteFaceSize == 0) {
            int height = grayFrame.rows();
            if (Math.round(height * 0.2f) > 0) {
                this.absoluteFaceSize = Math.round(height * 0.2f);
            }
        }

        // detect objects
        this.classifier.detectMultiScale(grayFrame, faces, 1.1, 2, Objdetect.CASCADE_SCALE_IMAGE,
                new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());

        // each rectangle in faces is a face: draw them!
        Rect[] objectsArray = faces.toArray();
        for (int i = 0; i < objectsArray.length; i++) {
            Imgproc.rectangle(frame, objectsArray[i].tl(), objectsArray[i].br(), new Scalar(0, 255, 0), 3);
        }

        // track the first face
        if(objectsArray.length != 0) {
            objectRoi = objectsArray[0];
        }

    }

    public Rect getObjectRoi(){
        return objectRoi;
    }
}