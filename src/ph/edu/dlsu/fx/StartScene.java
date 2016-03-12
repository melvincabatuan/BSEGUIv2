package ph.edu.dlsu.fx;

import javafx.scene.Parent;
import javafx.scene.image.ImageView;
import javafx.scene.layout.Pane;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import ph.edu.dlsu.fx.ui.CustomMenuItem;
import ph.edu.dlsu.fx.ui.MenuHBox;
import ph.edu.dlsu.fx.utils.Utils;
import ph.edu.dlsu.fx.vision.ConsensusMatchingTracker;
import ph.edu.dlsu.fx.vision.ObjectDetector;

/**
 * Created by cobalt on 3/6/16.
 */
public class StartScene extends BaseCameraScene {

    // cascade classifier
    private ObjectDetector faceDetector = new ObjectDetector();

    // tracker
    private ConsensusMatchingTracker cmt;
    private boolean isCmtInitialized;
    private Rect trackingRoi;

    // input
    private Mat mGray;

    private int initCounter;


    // Create content for the Main Menu scene
    public Parent createContent() {

        // Create the tracker
        cmt = new ConsensusMatchingTracker();
        isCmtInitialized = false;
        initCounter = 0;

        // Initialize gray image
        mGray = new Mat((int) frameHeight, (int) frameWidth, CvType.CV_8UC1);

        // Create Main Menu pane
        Pane rootNode = new Pane();
        rootNode.setPrefSize(displayWidth, displayHeight);

        // Initialize background image and load to Imageview
        ImageView imgBackground = Utils.loadImage2View("res/BSEdepth2.png", displayWidth, displayHeight);
        if (imgBackground != null) {
            rootNode.getChildren().add(imgBackground);
        }

        currentFrame = Utils.loadImage2View("res/video.jpg", frameWidth, frameHeight);
        currentFrame.setTranslateX((displayWidth - frameWidth) / 2.0);
        currentFrame.setTranslateY(0);
        rootNode.getChildren().add(currentFrame);
        startCamera();

        // Create Menu title and content
        createHMenu();

        // Add menu w/ title in the Pane
        rootNode.getChildren().add(menuBox);

        return rootNode;
    }

    public void createHMenu() {
        final CustomMenuItem home = new CustomMenuItem("HOME", menuWidth, menuHeight);
        final CustomMenuItem training = new CustomMenuItem("TRAINING", menuWidth, menuHeight);
        final CustomMenuItem facts = new CustomMenuItem("FACTS", menuWidth, menuHeight);
        final CustomMenuItem help = new CustomMenuItem("HELP", menuWidth, menuHeight);
        final CustomMenuItem about = new CustomMenuItem("ABOUT", menuWidth, menuHeight);
        final CustomMenuItem exit = new CustomMenuItem("EXIT", menuWidth, menuHeight);

        // handle menu events
        home.setOnMouseClicked(e -> {
            stopCamera();
            App.onHome();
        });

        training.setOnMouseClicked(e -> {
                    stopCamera();
                    App.onTutorial();
                }
        );

        facts.setOnMouseClicked(e -> {
            stopCamera();
            App.onFacts();
        });

        exit.setOnMouseClicked(e -> {
            Boolean confirmQuit = App.onExit();
            if (confirmQuit) {
                stopCamera();
            }
        });

        menuBox = new MenuHBox(
                home,
                training,
                facts,
                help,
                about,
                exit);

        menuBox.setTranslateX((displayWidth - 6 * menuWidth) / 2.0);
        menuBox.setTranslateY(0);
    }

    @Override
    public void onCameraFrame(Mat frame) {
        // get the gray image
        Imgproc.cvtColor(frame, mGray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(mGray, mGray);
        faceDetector.detectAndDisplay(frame);
        trackingRoi = faceDetector.getObjectRoi();
        if (!isCmtInitialized && trackingRoi != null) {
            initializeTracker();
        } else {
            cmt.apply(mGray, frame);
            initCounter++;
            // reinitialize every 5 frames
            if (initCounter > 5) {
                // reinitialize tracker if far from object detector roi
                initializeTracker();
                initCounter = 0;
            }
        }
    }

    private void initializeTracker() {
        cmt.initialize(mGray,
                (long) (trackingRoi.x),
                (long) (trackingRoi.y),
                (long) (trackingRoi.width),
                (long) (trackingRoi.height));
        isCmtInitialized = true;
    }

    @Override
    public void stopCamera() {
        mGray.release();
        cmt.release();
        trackingRoi = null;
        isCmtInitialized = false;
        super.stopCamera();
    }
}