package ph.edu.dlsu.fx.vision;

import org.opencv.core.Mat;
import ph.edu.dlsu.fx.vision.interfaces.NativeAlgorithm;

/**
 * Created by cobalt on 3/8/16.
 */
public final class RecolorRC implements NativeAlgorithm {

    static {
        // Load the native library if it is not already loaded.
        System.loadLibrary("mhealth_vision");
    }

    public RecolorRC(){
        mNativeAddr = nativeCreateObject();
    }

    @Override
    public void release() {
        nativeDestroyObject(mNativeAddr);
        mNativeAddr = 0;
    }

    @Override
    public void apply(final Mat src, final Mat dst) {
        apply(mNativeAddr, src.getNativeObjAddr(),
                dst.getNativeObjAddr());
    }

    // Ensure that release() is always called at least once
    // before the object is garbage-collected. This is calling
    // automatic memory management as a fallback if there is no
    // manual call to dispose.
    @Override
    protected void finalize() throws Throwable {
        release();
        super.finalize();
    }
    // Note: The finalize method is inherited from java.lang.Object
    // and is called when the object is garbage-collected.

    private long mNativeAddr = 0;

    private static native long nativeCreateObject();
    private static native void nativeDestroyObject(long thiz);
    private static native void apply(long thiz, long srcAddr, long dstAddr);
}