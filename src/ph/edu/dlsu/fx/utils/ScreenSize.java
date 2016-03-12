package ph.edu.dlsu.fx.utils;

import javafx.stage.Screen;

/**
 * Created by cobalt on 3/6/16.
 */
public class ScreenSize {

    public static double getDisplayWidth() {
        return Screen.getScreens().get(0).getBounds().getWidth();
    }

    public static double getDisplayHeight() {
        return Screen.getScreens().get(0).getBounds().getHeight();
    }
}
