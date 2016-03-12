package ph.edu.dlsu.fx.vision.interfaces;

import org.opencv.core.Mat;

/**
 * Created by cobalt on 3/8/16.
 */
public interface NativeAlgorithm {

    public abstract void apply(final Mat src, final Mat dst);

    public abstract void release();

}