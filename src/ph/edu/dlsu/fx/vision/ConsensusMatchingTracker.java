package ph.edu.dlsu.fx.vision;

import org.opencv.core.Mat;
import ph.edu.dlsu.fx.vision.interfaces.NativeAlgorithm;

/**
 * Created by cobalt on 3/9/16.
 */
public class ConsensusMatchingTracker implements NativeAlgorithm {

    static {
        // Load the native library if it is not already loaded.
        System.loadLibrary("mhealth_vision");
    }

    public ConsensusMatchingTracker() {
        mNativeAddr = nativeCreateObject();
    }


    public void initialize(final Mat srcGray, long xTopLeft, long yTopLeft, long width, long height) {
        initialize(mNativeAddr, srcGray.getNativeObjAddr(), xTopLeft, yTopLeft, width, height);
    }

    public void release() {
        nativeDestroyObject(mNativeAddr);
        mNativeAddr = 0;
    }


    public void apply(final Mat src, final Mat dst) {
        apply(mNativeAddr, src.getNativeObjAddr(),
                dst.getNativeObjAddr());
    }


    @Override
    protected void finalize() throws Throwable {
        release();
        super.finalize();
    }


    private long mNativeAddr = 0;

    private static native long nativeCreateObject();

    private static native void nativeDestroyObject(long thiz);

    private static native void initialize(long thiz, long srcAddr, long xTopLeft, long yTopLeft, long width, long height);

    private static native void apply(long thiz, long srcAddr, long dstAddr);

}