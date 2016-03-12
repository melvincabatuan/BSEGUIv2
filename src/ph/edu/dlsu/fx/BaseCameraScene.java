package ph.edu.dlsu.fx;

import javafx.scene.Parent;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import ph.edu.dlsu.fx.ui.MenuHBox;
import ph.edu.dlsu.fx.utils.ScreenSize;
import ph.edu.dlsu.fx.utils.Utils;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Created by cobalt on 3/6/16.
 */
public abstract class BaseCameraScene {

    MenuHBox menuBox;

    // Window size
    double displayWidth = ScreenSize.getDisplayWidth();
    double displayHeight = ScreenSize.getDisplayHeight();

    // Frame size
    double frameWidth = 0.8 * displayWidth;
    double frameHeight = displayHeight;

    // Menu size
    final double menuWidth = 220;
    final double menuHeight = 40;

    // display for image frames
    ImageView currentFrame;

    // a timer for acquiring the video stream
    ScheduledExecutorService timer;

    // the OpenCV object that realizes the video capture
    VideoCapture capture = new VideoCapture();


    // create content for the Main Menu scene
    public abstract Parent createContent();

    // create top menu
    public abstract void createHMenu();

    public abstract void onCameraFrame(Mat frame);

    public void startCamera() {

        // start the video capture
        this.capture.open(0);

        // is the video stream available?
        if (this.capture.isOpened()) {

            // grab a frame every 33 ms (30 frames/sec)
            Runnable frameGrabber = () -> {
                Image imageToShow = grabFrame();
                currentFrame.setImage(imageToShow);
            };

            this.timer = Executors.newSingleThreadScheduledExecutor();
            this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

        } else {
            System.err.println("Failed to open the camera...");
        }
    }

    public void stopCamera() {
        // stop the timer
        try {
            this.timer.shutdown();
            this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
        }

        // release the camera
        if (capture != null) {
            this.capture.release();
        }
        // clean the frame
        if (currentFrame != null) {
            this.currentFrame.setImage(null);
        }
    }


    private Image grabFrame() {

        Image imageToShow = null;
        Mat frame = new Mat();

        // check if the capture is open
        if (this.capture.isOpened()) {
            try {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {

                    // process frame here:
                    onCameraFrame(frame);

                    // convert the Mat object (OpenCV) to Image (JavaFX)
                    imageToShow = Utils.mat2Image(frame);
                }

            } catch (Exception e) {
                System.err.println("Exception during the image elaboration: " + e);
            }
        }
        return imageToShow;
    }
}