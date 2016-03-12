package ph.edu.dlsu.fx.utils;

import javafx.scene.media.AudioClip;

/**
 * Created by cobalt on 3/6/16.
 */
public class Sound {

    private AudioClip menuClip;

    public Sound(String path) {
        try {
            menuClip = new AudioClip(path);
        } catch (Exception e) {
            System.out.println("Failed to load audio clip!");
        }
    }

    public void play() {
        menuClip.play();
    }
}