package ph.edu.dlsu.fx.utils;

import javafx.animation.FadeTransition;
import javafx.animation.TranslateTransition;
import javafx.application.Platform;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.ReadOnlyDoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.geometry.Point2D;
import javafx.geometry.Pos;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Arc;
import javafx.scene.shape.ArcType;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.stage.WindowEvent;
import javafx.util.Duration;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Created by cobalt on 3/12/16.
 */
public class ImageBox {

    private static final String photoViewerStyle = "/home/cobalt/IdeaProjects/BSEGUIv2/src/ph/edu/dlsu/fx/css/photo-viewer.css";
    private static final String testImagePath = "/home/cobalt/IdeaProjects/BSEGUIv2/res/drawable/fruitsveggie.jpeg";
    private static final String CLOSE_BUTTON_ID = "close-button";

    // List of URL strings
    private static final List<String> imageFiles = new ArrayList<>();

    // The current index into the imageFile
    private static int currentIndex = -1;

    // Enumeration of next and previous button directions
    public enum ButtonMove {
        NEXT, PREV
    }

    ;

    // Image caption animation
    private static Group caption;

    // Current image view display
    private static ImageView currentImageView;

    // mutex */
    private static AtomicBoolean loading = new AtomicBoolean();

    private static Stage stage;

    private static Point2D anchorPt;
    private static Point2D previousLocation;


    public static void show() {

        stage = new Stage();
        stage.initModality(Modality.APPLICATION_MODAL);
        stage.initStyle(StageStyle.TRANSPARENT);
        stage.setX(400);
        stage.setY(120);

        double scale = 1.5;
        double sceneWidth = 640.0 * scale;
        double sceneHeight = 360.0 * scale;

        Group root = new Group();
        Scene scene = new Scene(root, sceneWidth, sceneHeight, Color.BLACK);
        scene.setFill(null);

        // Load JavaFX CSS styles
        scene.getStylesheets().add("file://" + photoViewerStyle);
        stage.setScene(scene);

        initFullScreenMode();
        initMovableWindow();
        initializeImages();

        // set up the current image view area
        currentImageView = createImageView(scene.heightProperty());

        // center image
        StackPane imagePane = new StackPane();
        imagePane.setPrefSize(sceneWidth, sceneHeight);
        imagePane.getChildren().add(currentImageView);
        imagePane.maxWidthProperty().bind(scene.widthProperty());
        imagePane.minWidthProperty().bind(scene.widthProperty());
        imagePane.maxHeightProperty().bind(scene.heightProperty());
        imagePane.minHeightProperty().bind(scene.heightProperty());
        imagePane.setAlignment(Pos.CENTER);

        // create button panel controls (left & right arrows)
        Group buttonGroup = createButtonPanel(scene);

        // Add ticker
        caption = createTickerControl(stage, 85); // 85: constant adjusts right padding

        // Create the close button
        Node closeButton = createCloseButton();

        // Add nodes to the group
        root.getChildren().addAll(imagePane,
                buttonGroup,
                caption,
                closeButton);

        stage.setFullScreen(true);
        stage.show();
    }


    private static void initFullScreenMode() {
        Scene scene = stage.getScene();
        scene.setOnMouseClicked((MouseEvent event) -> {
            if (event.getClickCount() == 2) {
                stage.setFullScreen(!stage.isFullScreen());
            }
        });
    }


    private static void initMovableWindow() {
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


    private static void initializeImages() {
        try {
            addImage("file://" + testImagePath);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        if (currentIndex > -1) {
            loadImage(imageFiles.get(currentIndex));
        }
    }


    private static Group createButtonPanel(Scene scene) {
        // create button panel
        Group buttonGroup = new Group();
        Rectangle buttonArea = new Rectangle(0, 0, 65, 30);
        buttonArea.getStyleClass().add("button-panel");
        buttonGroup.getChildren().add(buttonArea);
        // left arrow button
        Arc leftButton = new Arc(12, 16, 15, 15, -30, 60);
        leftButton.setType(ArcType.ROUND);
        leftButton.getStyleClass().add("left-arrow");

        // return to previous image
        leftButton.addEventHandler(MouseEvent.MOUSE_PRESSED,
                (mouseEvent) -> {
                    if (currentIndex == 0 || loading.get()) return;
                    int indx = gotoImageIndex(ButtonMove.PREV);
                    if (indx > -1) {
                        loadImage(imageFiles.get(indx));
                    }
                });

        // right arrow button
        Arc rightButton = new Arc(12, 16, 15, 15, 180 - 30, 60);
        rightButton.setType(ArcType.ROUND);
        rightButton.getStyleClass().add("right-arrow");
        // advance to next image
        rightButton.addEventHandler(MouseEvent.MOUSE_PRESSED,
                (mouseEvent) -> {
                    // if no next image or currently loading.
                    if (currentIndex == imageFiles.size() - 1
                            || loading.get()) return;
                    int indx = gotoImageIndex(ButtonMove.NEXT);
                    if (indx > -1) {
                        loadImage(imageFiles.get(indx));
                    }
                });

        // add buttons to button group
        buttonGroup.getChildren().addAll(leftButton, rightButton);
        // move button group when scene is resized
        buttonGroup.translateXProperty()
                .bind(scene.widthProperty()
                        .subtract(buttonArea.getWidth() + 6));
        buttonGroup.translateYProperty()
                .bind(scene.heightProperty()
                        .subtract(buttonArea.getHeight() + 6));


        // Fade in button controls
        scene.setOnMouseEntered((MouseEvent me) -> {
            FadeTransition fadeButtons =
                    new FadeTransition(Duration.millis(500), buttonGroup);
            fadeButtons.setFromValue(0.0);
            fadeButtons.setToValue(1.0);
            fadeButtons.play();
        });

        // Fade out button controls
        scene.setOnMouseExited((MouseEvent me) -> {
            FadeTransition fadeButtons =
                    new FadeTransition(Duration.millis(500), buttonGroup);
            fadeButtons.setFromValue(1);
            fadeButtons.setToValue(0);
            fadeButtons.play();
        });

        return buttonGroup;
    }


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


    private static ImageView createImageView(ReadOnlyDoubleProperty heightProperty) {
        ImageView imageView = new ImageView();
        imageView.setPreserveRatio(true);
        imageView.fitHeightProperty().bind(heightProperty);
        return imageView;
    }


    private static boolean isValidImageFile(String url) {
        List<String> imgTypes = Arrays.asList(".jpg", ".jpeg", ".png", ".gif", ".bmp");
        return imgTypes.stream()
                .anyMatch(url::endsWith);
    }


    private static void addImage(String url) {
        if (isValidImageFile(url)) {
            currentIndex += 1;
            imageFiles.add(currentIndex, url);
        }
    }


    private static int gotoImageIndex(ButtonMove direction) {
        int size = imageFiles.size();
        if (size == 0) {
            currentIndex = -1;
        } else if (direction == ButtonMove.NEXT
                && size > 1
                && currentIndex < size - 1) {
            currentIndex += 1;
        } else if (direction == ButtonMove.PREV
                && size > 1
                && currentIndex > 0) {
            currentIndex -= 1;
        }
        return currentIndex;
    }


    private static Task createWorker(final String url) {
        return new Task() {
            @Override
            protected Object call() throws Exception {
                // on the worker thread...
                Image image = new Image(url, false);
                Platform.runLater(() -> {
                    // on the JavaFX Application Thread...."done loading image "
                    currentImageView.setImage(image);
                    loading.set(false); // free lock
                });
                return true;
            }
        };
    }


    private static void loadImage(String url) {
        if (!loading.getAndSet(true)) {
            Task loadImage = createWorker(url);
            new Thread(loadImage).start();
        }
    }


    private static Group createTickerControl(Stage stage, double rightPadding) {
        Scene scene = stage.getScene();

        // create ticker area
        Group tickerArea = new Group();
        Rectangle tickerRect = new Rectangle(scene.getWidth(), 38); // constant height
        tickerRect.getStyleClass().add("ticker-border");
        tickerRect.setFill(Color.BLACK);
        tickerRect.setVisible(false);

        Rectangle clipRegion = new Rectangle(scene.getWidth(), 38);
        clipRegion.getStyleClass().add("ticker-clip-region");
        tickerArea.setClip(clipRegion);

        // Resize the ticker area when the window is resized
        tickerArea.setTranslateX(6);
        tickerArea.translateYProperty()
                .bind(scene.heightProperty()
                        .subtract(tickerRect.getHeight() + 6));
        tickerRect.widthProperty()
                .bind(scene.widthProperty()
                        .subtract(rightPadding));
        clipRegion.widthProperty()
                .bind(scene.widthProperty()
                        .subtract(rightPadding));
        tickerArea.getChildren().add(tickerRect);

        // News feed container
        FlowPane tickerContent = new FlowPane();

        Text news = new Text();
        news.setText("JavaFX 8.0 News! | 85 and sunny | :)");
        news.setFill(Color.DARKGREY);
        news.setFont(Font.font("Times New Roman", FontWeight.SEMI_BOLD, 30));

        tickerContent.getChildren().add(news);
        DoubleProperty centerContentY = new SimpleDoubleProperty();
        centerContentY.bind(
                clipRegion.heightProperty()
                        .divide(2)
                        .subtract(tickerContent.heightProperty()
                                .divide(2)));
        tickerContent.translateYProperty().bind(centerContentY);
        tickerArea.getChildren().add(tickerContent);

        // scroll news feed
        TranslateTransition tickerScroller = new TranslateTransition();
        tickerScroller.setNode(tickerContent);
        tickerScroller.setDuration(
                Duration.millis(scene.getWidth() * 40));
        tickerScroller.fromXProperty()
                .bind(scene.widthProperty());
        tickerScroller.toXProperty()
                .bind(tickerContent.widthProperty()
                        .negate());

        // when ticker has finished, reset and replay ticker animation
        tickerScroller.setOnFinished((ActionEvent ae) -> {
            tickerScroller.stop();
            tickerScroller.setDuration(
                    Duration.millis(scene.getWidth() * 40));
            tickerScroller.playFromStart();
        });

        // start ticker after nodes are shown
        stage.setOnShown(windowEvent -> tickerScroller.play());
        return tickerArea;
    }
}