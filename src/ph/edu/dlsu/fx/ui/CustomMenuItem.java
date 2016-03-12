package ph.edu.dlsu.fx.ui;

import javafx.geometry.Pos;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.paint.CycleMethod;
import javafx.scene.paint.LinearGradient;
import javafx.scene.paint.Stop;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import ph.edu.dlsu.fx.utils.Sound;

/**
 * Created by cobalt on 3/6/16.
 */
public class CustomMenuItem extends StackPane {

    // private static final String menuClickPath = "/home/cobalt/IdeaProjects/BSEGUIv2/res/audio/menuclick.wav";

    public CustomMenuItem(String name) {
        LinearGradient gradient = new LinearGradient(0, 0, 1, 0, true, CycleMethod.NO_CYCLE, new Stop(0, Color.DARKOLIVEGREEN),
                new Stop(0.1, Color.BLACK),
                new Stop(0.9, Color.BLACK),
                new Stop(1, Color.DARKOLIVEGREEN));

        Rectangle bg = new Rectangle(240, 40);
        bg.setOpacity(0.4);

        Text text = new Text(name);
        text.setFill(Color.DARKGREY);
        text.setFont(Font.font("Times New Roman", FontWeight.SEMI_BOLD, 30));

        setAlignment(Pos.CENTER);

        getChildren().addAll(bg, text);

        setOnMouseEntered(event -> {
            bg.setFill(gradient);
            text.setFill(Color.WHITE);

        });

        setOnMouseExited(event -> {
            bg.setFill(Color.BLACK);
            text.setFill(Color.DARKGREY);
        });

        setOnMousePressed(event ->
        {
            bg.setFill(Color.DARKGREEN);
            (new Sound("file:res/audio/menuclick.wav")).play();
        });

        setOnMouseReleased(event -> bg.setFill(gradient));

    }


    public CustomMenuItem(String name, double rectWidth, double rectHeight) {
        LinearGradient gradient = new LinearGradient(0, 0, 1, 0, true, CycleMethod.NO_CYCLE, new Stop(0, Color.DARKOLIVEGREEN),
                new Stop(0.1, Color.BLACK),
                new Stop(0.9, Color.BLACK),
                new Stop(1, Color.DARKOLIVEGREEN));

        Rectangle bg = new Rectangle(rectWidth, rectHeight);
        bg.setOpacity(0.4);

        Text text = new Text(name);
        text.setFill(Color.DARKGREY);
        text.setFont(Font.font("Times New Roman", FontWeight.SEMI_BOLD, 30));

        setAlignment(Pos.CENTER);

        getChildren().addAll(bg, text);

        setOnMouseEntered(event -> {
            bg.setFill(gradient);
            text.setFill(Color.WHITE);

        });

        setOnMouseExited(event -> {
            bg.setFill(Color.BLACK);
            text.setFill(Color.DARKGREY);
        });

        setOnMousePressed(event ->
        {
            bg.setFill(Color.DARKGREEN);
            (new Sound("file:res/audio/menuclick.wav")).play();
        });

        setOnMouseReleased(event -> bg.setFill(gradient));

    }
}