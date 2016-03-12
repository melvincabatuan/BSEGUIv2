package ph.edu.dlsu.fx.utils;

import javafx.beans.value.ChangeListener;
import javafx.geometry.Point2D;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.Slider;
import javafx.scene.input.MouseEvent;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.MediaView;
import javafx.scene.paint.Color;
import javafx.scene.shape.*;
import javafx.scene.text.Text;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.stage.WindowEvent;
import javafx.util.Duration;

/**
 * Created by cobalt on 3/12/16.
 */
public class VideoBox {

    private static final String buttonPanelStyle = "/home/cobalt/IdeaProjects/BSEGUIv2/src/ph/edu/dlsu/fx/css/playing-video.css";
    private static final String MEDIA_VIEW_ID = "media-view";
    private static final String STOP_BUTTON_ID = "stop-button";
    private static final String PLAY_BUTTON_ID = "play-button";
    private static final String PAUSE_BUTTON_ID = "pause-button";
    private static final String CLOSE_BUTTON_ID = "close-button";
    private static final String SEEK_POS_SLIDER_ID = "seek-position-slider";
    private static Point2D anchorPt;
    private static Point2D previousLocation;
    private static MediaPlayer mediaPlayer;
    private static ChangeListener<Duration> progressListener;
    private static Stage stage;



    public static void show(String url) {
        stage = new Stage();
        stage.initModality(Modality.APPLICATION_MODAL);
        stage.initStyle(StageStyle.TRANSPARENT);
        //stage.setFullScreen(true);
        stage.setX(400);
        stage.setY(120);

        double scale = 1.5;

        Group root = new Group();
        Scene scene = new Scene(root, 640.0 * scale, 360.0 * scale, Color.rgb(0, 0, 0, 0));

        // Load JavaFX CSS style
        scene.getStylesheets().add("file://" + buttonPanelStyle);
        stage.setScene(scene);

        // Initialize stage to be movable via mouse
        initMovablePlayer();

        // Initialize stage to have fullscreen ability
        initFullScreenMode();

        // Create a media view to display video
        MediaView mediaView = createMediaView();

        // application area
        Node applicationArea = createApplicationArea();

        // Create the button panel
        Node buttonPanel = createButtonPanel();

        // Progress and seek position slider
        Slider progressSlider = createSlider();

        // Update slider as video is progressing
        progressListener = (observable, oldValue, newValue) ->
                progressSlider.setValue(newValue.toSeconds());


        // Create the close button
        Node closeButton = createCloseButton();
        root.getChildren()
                .addAll(applicationArea,
                        mediaView,
                        buttonPanel,
                        progressSlider,
                        closeButton);

        // Play video file
        try {
            playMedia(url);
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        stage.show();
    }

    private static void initFullScreenMode() {
        Scene scene = stage.getScene();
// Full screen toggle
        scene.setOnMouseClicked((MouseEvent event) -> {
            if (event.getClickCount() == 2) {
                stage.setFullScreen(!stage.isFullScreen());
            }
        });
    }


    /**
     * Initialize the stage to allow the mouse cursor to
     * move the application using dragging.
     */
    private static void initMovablePlayer() {
        Scene scene = stage.getScene();
        // starting initial anchor point
        scene.setOnMousePressed(mouseEvent
                -> anchorPt = new Point2D(mouseEvent.getScreenX(),
                mouseEvent.getScreenY())
        );
        // dragging the entire stage
        scene.setOnMouseDragged(mouseEvent -> {
            if (anchorPt != null && previousLocation != null) {
                stage.setX(previousLocation.getX()
                        + mouseEvent.getScreenX()
                        - anchorPt.getX());
                stage.setY(previousLocation.getY()
                        + mouseEvent.getScreenY()
                        - anchorPt.getY());
            }
        });
        // set the current location
        scene.setOnMouseReleased(mouseEvent
                -> previousLocation = new Point2D(stage.getX(),
                stage.getY())
        );
        // Initialize previousLocation after Stage is shown
        stage.addEventHandler(WindowEvent.WINDOW_SHOWN,
                (WindowEvent t) -> previousLocation = new Point2D(stage.getX(),
                        stage.getY()));
    }


    /**
     * A simple rectangular area as the surface of the app.
     *
     * @return Node a Rectangle node.
     */
    private static Node createApplicationArea() {
        Scene scene = stage.getScene();
        Rectangle applicationArea = new Rectangle();
        // add selector to style app-area
        applicationArea.setId("app-area");
        // make the app area rectangle the size of the scene.
        applicationArea.widthProperty()
                .bind(scene.widthProperty());
        applicationArea.heightProperty()
                .bind(scene.heightProperty());
        return applicationArea;
    }


    /**
     * Creates a node containing the audio player's
     * stop, pause and play buttons.
     *
     * @return Node A button panel having play,
     * pause and stop buttons.
     */
    private static Node createButtonPanel() {
        Scene scene = stage.getScene();
        // create button control panel
        Group buttonGroup = new Group();
        // Button area
        Rectangle buttonArea = new Rectangle(60, 30);
        buttonArea.setId("button-area");
        buttonGroup.getChildren()
                .add(buttonArea);
        // stop button control
        Node stopButton = new Rectangle(10, 10);
        stopButton.setId(STOP_BUTTON_ID);
        stopButton.setOnMousePressed(mouseEvent -> {
            if (mediaPlayer != null) {
                updatePlayAndPauseButtons(true);
                if (mediaPlayer.getStatus() == MediaPlayer.Status.PLAYING) {
                    mediaPlayer.stop();
                }
            }
        }); // setOnMousePressed()

        // play button
        Arc playButton = new Arc(12, // center x
                16, // center y
                15, // radius x
                15, // radius y
                150, // start angle
                60); // length
        playButton.setId(PLAY_BUTTON_ID);
        playButton.setType(ArcType.ROUND);
        playButton.setOnMousePressed(mouseEvent -> mediaPlayer.play());
        // pause control
        Group pauseButton = new Group();
        pauseButton.setId(PAUSE_BUTTON_ID);
        Node pauseBackground = new Circle(12, 16, 10);
        pauseBackground.getStyleClass().add("pause-circle");
        Node firstLine = new Line(6, // start x
                6, // start y
                6, // end x
                14); // end y
        firstLine.getStyleClass()
                .add("pause-line");
        firstLine.setStyle("-fx-translate-x: 34;");
        Node secondLine = new Line(6, // start x
                6, // start y
                6, // end x
                14); // end y
        secondLine.getStyleClass().add("pause-line");
        secondLine.setStyle("-fx-translate-x: 38;");
        pauseButton.getChildren()
                .addAll(pauseBackground, firstLine, secondLine);
        pauseButton.setOnMousePressed(mouseEvent -> {
            if (mediaPlayer != null) {
                updatePlayAndPauseButtons(true);
                if (mediaPlayer.getStatus() == MediaPlayer.Status.PLAYING) {
                    mediaPlayer.pause();
                }
            }
        });
        // setOnMousePressed()
        playButton.setOnMousePressed(mouseEvent -> {
            if (mediaPlayer != null) {
                updatePlayAndPauseButtons(false);
                mediaPlayer.play();
            }
        });
        // setOnMousePressed()
        buttonGroup.getChildren()
                .addAll(stopButton,
                        playButton,
                        pauseButton);

        // move button group when scene is resized
        buttonGroup.translateXProperty()
                .bind(scene.widthProperty()
                        .subtract(buttonArea.getWidth() + 6));
        buttonGroup.translateYProperty()
                .bind(scene.heightProperty()
                        .subtract(buttonArea.getHeight() + 6));
        return buttonGroup;
    }


    /**
     * The close button to exit application
     *
     * @return Node representing a close button.
     */
    private static Node createCloseButton() {
        Scene scene = stage.getScene();
        Group closeButton = new Group();
        closeButton.setId(CLOSE_BUTTON_ID);
        Node closeBackground = new Circle(6, 0, 8);
        closeBackground.setId("close-circle");
        Node closeXmark = new Text(2, 4, "X");
        closeButton.translateXProperty()
                .bind(scene.widthProperty()
                        .subtract(15));
        closeButton.setTranslateY(10);
        closeButton.getChildren()
                .addAll(closeBackground, closeXmark);
        // go to home menu
        closeButton.setOnMouseClicked(mouseEvent -> stage.close());
        return closeButton;
    }

    /**
     * After a file is dragged onto the application a new MediaPlayer
     * instance is created with a media file.
     *
     * @param url The URL pointing to an audio file
     */
    private static void playMedia(String url) {
        Scene scene = stage.getScene();
        if (mediaPlayer != null) {
            mediaPlayer.pause();
            mediaPlayer.setOnPaused(null);
            mediaPlayer.setOnPlaying(null);
            mediaPlayer.setOnReady(null);
            mediaPlayer.currentTimeProperty()
                    .removeListener(progressListener);
            mediaPlayer.setAudioSpectrumListener(null);
        }

        Media media = new Media(url);

        // display media's metadata
        for (String s : media.getMetadata().keySet()) {
            System.out.println(s);
        }

        mediaPlayer = new MediaPlayer(media);
        // as the media is playing move the slider for progress
        mediaPlayer.currentTimeProperty()
                .addListener(progressListener);
        mediaPlayer.setOnReady(() -> {
            updatePlayAndPauseButtons(false);
            Slider progressSlider =
                    (Slider) scene.lookup("#" + SEEK_POS_SLIDER_ID);
            progressSlider.setValue(0);
            progressSlider.setMax(mediaPlayer.getMedia()
                    .getDuration()
                    .toSeconds());
            mediaPlayer.play();
        }); // setOnReady()
// back to the beginning
        mediaPlayer.setOnEndOfMedia(() -> {
            updatePlayAndPauseButtons(true);
// change buttons to play and rewind
            mediaPlayer.stop();
        }); // setOnEndOfMedia()

        // set the media player to display video
        MediaView mediaView = (MediaView) scene.lookup("#" + MEDIA_VIEW_ID);
        mediaView.setMediaPlayer(mediaPlayer);
    }


    /**
     * Create a MediaView node.
     *
     * @return MediaView
     */
    private static MediaView createMediaView() {
        MediaView mediaView = new MediaView();
        mediaView.setId(MEDIA_VIEW_ID);
        //mediaView.setPreserveRatio(true);
        mediaView.setSmooth(true);

        mediaView.fitWidthProperty()
                .bind(stage.getScene()
                        .widthProperty());
        mediaView.fitHeightProperty()
                .bind(stage.getScene()
                        .heightProperty());

        mediaView.setOnError(mediaErrorEvent -> mediaErrorEvent.getMediaError()
                .printStackTrace());
        return mediaView;
    }


    /**
     * Sets play button visible and pause button not visible when
     * playVisible is true otherwise the opposite.
     * <p>
     * * @param playVisible - value of true the play becomes visible
     * and pause non visible, otherwise the opposite.
     */
    private static void updatePlayAndPauseButtons(boolean playVisible) {
        Scene scene = stage.getScene();
        Node playButton = scene.lookup("#" + PLAY_BUTTON_ID);
        Node pauseButton = scene.lookup("#" + PAUSE_BUTTON_ID);
// hide or show buttons
        playButton.setVisible(playVisible);
        pauseButton.setVisible(!playVisible);
        if (playVisible) {
// show play button
            playButton.toFront();
            pauseButton.toBack();
        } else {
// show pause button
            pauseButton.toFront();
            playButton.toBack();
        }
    }

    /**
     * A position slider to seek backward and forward
     * that is bound to a media player control.
     *
     * @return Slider control bound to media player.
     */
    private static Slider createSlider() {
        Slider slider = new Slider(0, 100, 1);
        slider.setId(SEEK_POS_SLIDER_ID);
        slider.valueProperty()
                .addListener((observable) -> {
                    if (slider.isValueChanging()) {
                        // must check if media is paused before seeking
                        if (mediaPlayer != null &&
                                mediaPlayer.getStatus() == MediaPlayer.Status.PAUSED) {
                            // convert seconds to millis
                            double dur = slider.getValue() * 1000;
                            mediaPlayer.seek(Duration.millis(dur));
                        }
                    }
                }); //addListener()
        Scene scene = stage.getScene();
        slider.setTranslateX(10);
        slider.translateYProperty()
                .bind(scene.heightProperty()
                        .subtract(30));
        return slider;
    }
}