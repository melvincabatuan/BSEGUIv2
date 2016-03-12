package ph.edu.dlsu.fx.ui;

import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Line;

/**
 * Created by cobalt on 3/6/16.
 */
public class MenuVBox extends VBox {



    public MenuVBox(CustomMenuItem... items) {
        getChildren().add(createSeparator());

        for (CustomMenuItem item : items) {
            getChildren().addAll(item, createSeparator());
        }
    }

    private Line createSeparator() {
        Line sep = new Line();
        sep.setEndX(240);
        sep.setStroke(Color.DARKGREY);
        return sep;
    }

}