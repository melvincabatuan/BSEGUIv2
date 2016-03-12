package ph.edu.dlsu.fx.ui;

import javafx.geometry.Pos;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;

/**
 * Created by cobalt on 3/6/16.
 */
public class MenuTitle extends StackPane {
    public MenuTitle(String name) {
        Rectangle bg = new Rectangle(336, 60);
        bg.setStroke(Color.WHITE);
        bg.setStrokeWidth(2);
        bg.setFill(null);

        Text text = new Text(name);
        text.setFill(Color.WHITE);
        text.setFont(Font.font("Times New Roman", FontWeight.SEMI_BOLD, 50));

        setAlignment(Pos.CENTER_LEFT);
        getChildren().addAll(bg, text);
    }
}