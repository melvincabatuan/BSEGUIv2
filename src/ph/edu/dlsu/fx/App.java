package ph.edu.dlsu.fx;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.ImageView;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;
import org.opencv.core.Core;
import ph.edu.dlsu.fx.ui.CustomMenuItem;
import ph.edu.dlsu.fx.ui.MenuTitle;
import ph.edu.dlsu.fx.ui.MenuVBox;
import ph.edu.dlsu.fx.utils.*;

import java.nio.file.Paths;

public class App extends Application {

    private static final String introFilePath = "/home/cobalt/IdeaProjects/BSEGUIv2/res/video/GameofThronesTheme.mp4";
    private static final String WINDOW_TITLE = "BSE APPLICATION -- Alpha Version";
    public static final String MENU_TITLE    = "    BSE MENU";

    // Window size
    private static double displayWidth;
    private static double displayHeight;

    // App stage
    static Stage stage;

    // Main Menu
    MenuVBox menuBox;

    // Scene
    static Scene menuScene;


    @Override
    public void start(Stage primaryStage) throws Exception {
        initializeScreenSize();
        menuScene = new Scene(createHomeContent());
        stage = primaryStage;
        stage.setTitle(WINDOW_TITLE);
        stage.setScene(menuScene);
        stage.setMaximized(true);
        //stage.setFullScreen(true);
        stage.show();
    }

    private void initializeScreenSize() {
        displayWidth = ScreenSize.getDisplayWidth();
        displayHeight = ScreenSize.getDisplayHeight();
    }


    // Create content for the Main Menu scene
    private Parent createHomeContent() {

        // Create Main Menu pane
        Pane rootNode = new Pane();
        rootNode.setPrefSize(displayWidth, displayHeight);

        // Initialize background image and load to Imageview
        ImageView imgBackground = Utils.loadImage2View("res/drawable/skyrim.jpg", displayWidth, displayHeight);
        if (imgBackground != null) {
            rootNode.getChildren().add(imgBackground);
        }

        // Create Menu title and content
        MenuTitle title = new MenuTitle(MENU_TITLE);
        title.setTranslateX(50);
        title.setTranslateY(200);
        createVMenu();

        // Add menu w/ title in the Pane
        rootNode.getChildren().addAll(title, menuBox);
        return rootNode;
    }


    private void createVMenu() {

        final CustomMenuItem intro =    new CustomMenuItem("INTRO");
        final CustomMenuItem facts =    new CustomMenuItem("FACTS");
        final CustomMenuItem tutorial = new CustomMenuItem("TUTORIAL");
        final CustomMenuItem visual =   new CustomMenuItem("VISUAL");
        final CustomMenuItem tactile =  new CustomMenuItem("TACTILE");
        final CustomMenuItem help =     new CustomMenuItem("HELP");
        final CustomMenuItem exit =     new CustomMenuItem("EXIT");

        // handle menu events
        intro.setOnMouseClicked(e -> onIntro());
        facts.setOnMouseClicked(e -> onFacts());
        tutorial.setOnMouseClicked(e -> onTutorial());
        visual.setOnMouseClicked(e -> onVisual());
        tactile.setOnMouseClicked(e -> onTactile());
        help.setOnMouseClicked(e -> onHelp());
        exit.setOnMouseClicked(e -> onExit());

        menuBox = new MenuVBox(
                intro,
                facts,
                tutorial,
                visual,
                tactile,
                help,
                exit);

        menuBox.setTranslateX(100);
        menuBox.setTranslateY(300);
    }



    // HOME Menu
    public static void onHome() {
        stage.setTitle(WINDOW_TITLE);
        stage.setScene(menuScene);
    }

    // INTRO Menu
    public static void onIntro() {
        try {
            //String url = Paths.get(filePath).toUri().toURL().toString(); // ok
            String url = "file://" + introFilePath;
            VideoBox.show(url);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    // FACTS Menu
    public static void onFacts() {
        ImageBox.show();
    }


    // TUTORIAL Menu
    public static void onTutorial() {
        try {
            String url = Paths.get(introFilePath).toUri().toURL().toString();
            VideoBox.show(url);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private void onVisual() {
        FactsScene factScene = new FactsScene();
        stage.setScene(new Scene(factScene.createContent(), displayWidth, displayHeight));

    }

    private void onTactile() {
        FactsScene factScene = new FactsScene();
        stage.setScene(new Scene(factScene.createContent(), displayWidth, displayHeight));
    }


    private void onHelp() {

    }


    // EXIT Menu
    public static boolean onExit() {
        boolean confirmQuit = ConfirmationBox.show(
                "Are you sure you want to quit?",
                "Yes", "No");
        if (confirmQuit) {
            // Perform cleanup tasks here
            Platform.exit();
        }
        return confirmQuit;
    }


    // Load OpenCV in main()
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME); // OpenCV
        launch(args);
    }
}