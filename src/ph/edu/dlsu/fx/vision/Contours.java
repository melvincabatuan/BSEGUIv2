package ph.edu.dlsu.fx.vision;

import org.opencv.core.Mat;
import ph.edu.dlsu.fx.vision.interfaces.NativeAlgorithm;

/**
 * Created by cobalt on 3/8/16.
 */
public final class Contours implements NativeAlgorithm {

    static {
        // Load the native library if it is not already loaded.
        System.loadLibrary("mhealth_vision");
    }

    public Contours(){
        mNativeAddr = nativeCreateObject();
    }

    @Override
    public void apply(Mat src, Mat dst) {
        apply(mNativeAddr, src.getNativeObjAddr(),
                dst.getNativeObjAddr());
    }

    @Override
    public void release() {
        nativeDestroyObject(mNativeAddr);
        mNativeAddr = 0;
    }

    @Override
    protected void finalize() throws Throwable {
        release();
        super.finalize();
    }

    private long mNativeAddr = 0;

    private static native long nativeCreateObject();
    private static native void nativeDestroyObject(long thiz);
    private static native void apply(long thiz, long srcAddr, long dstAddr);
}