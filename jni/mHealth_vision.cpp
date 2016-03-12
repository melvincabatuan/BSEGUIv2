//
// Created by cobalt on 1/4/16.
// mHealth JNI
//

#include <jni.h>
#include <opencv2/core.hpp>

/* Mixers (Filter)*/
#include "RecolorRC.h"

/* Contours */
#include "Contours.h"

/* CMT Tracker */
#include "ConsensusMatchingTracker.h"


using namespace mhealth;


#ifdef __cplusplus
extern "C" {
#endif


/****************************** Consensus-based Matching Tracker ******************************/

JNIEXPORT jlong JNICALL
Java_ph_edu_dlsu_fx_vision_ConsensusMatchingTracker_nativeCreateObject(JNIEnv *env,
                                                                            jclass type){
    ConsensusMatchingTracker *self = new ConsensusMatchingTracker();
    return (jlong) self;

}

JNIEXPORT void JNICALL
Java_ph_edu_dlsu_fx_vision_ConsensusMatchingTracker_nativeDestroyObject(JNIEnv *env,
                                                                             jclass type,
                                                                             jlong thiz) {

    if (thiz != 0) {
        ConsensusMatchingTracker *self = (ConsensusMatchingTracker *) thiz;
        delete self;
    }

}

JNIEXPORT void JNICALL
Java_ph_edu_dlsu_fx_vision_ConsensusMatchingTracker_initialize(JNIEnv *env,
                                                                            jclass type, jlong thiz,
                                                                            jlong srcAddr,
                                                                            jlong xTopLeft,
                                                                            jlong yTopLeft,
                                                                            jlong width,
                                                                            jlong height) {

    ConsensusMatchingTracker *self = (ConsensusMatchingTracker *) thiz;
    cv::Mat& im_gray  = *(cv::Mat*)srcAddr;
    self->initialize(im_gray, xTopLeft, yTopLeft, width, height);

}

JNIEXPORT void JNICALL
Java_ph_edu_dlsu_fx_vision_ConsensusMatchingTracker_apply(JNIEnv *env, jclass type,
                                                                    jlong thiz, jlong srcAddr,
                                                                    jlong dstAddr) {

    ConsensusMatchingTracker *self = (ConsensusMatchingTracker *) thiz;

    if (!(self->isInitialized()))
        return;

    cv::Mat& im_gray  = *(cv::Mat*)srcAddr;
    cv::Mat& im_rgba  = *(cv::Mat*)dstAddr;

    self->processFrame(im_gray, im_rgba);

}



/****************************** RecolorRC ******************************/

JNIEXPORT jlong JNICALL Java_ph_edu_dlsu_fx_vision_RecolorRC_nativeCreateObject(JNIEnv *env, jclass type) {

    RecolorRCFilter *self = new RecolorRCFilter();
    return (jlong) self;

}


JNIEXPORT void JNICALL Java_ph_edu_dlsu_fx_vision_RecolorRC_nativeDestroyObject(JNIEnv *env, jclass type,
                                                              jlong thiz) {

    if (thiz != 0) {
        RecolorRCFilter *self = (RecolorRCFilter *) thiz;
        delete self;
    }

}

JNIEXPORT void JNICALL Java_ph_edu_dlsu_fx_vision_RecolorRC_apply(JNIEnv *env, jclass type, jlong thiz,
                                                     jlong srcAddr, jlong dstAddr) {

    if (thiz != 0) {
        RecolorRCFilter *self = (RecolorRCFilter *) thiz;
        cv::Mat &src = *(cv::Mat *) srcAddr;
        cv::Mat &dst = *(cv::Mat *) dstAddr;
        self->apply(src, dst);
    }

}


/******************************** Contours *******************************************/

JNIEXPORT jlong JNICALL Java_ph_edu_dlsu_fx_vision_Contours_nativeCreateObject
  (JNIEnv *env, jclass type){
      Contours *self = new Contours();
      return (jlong) self;
  }

JNIEXPORT void JNICALL Java_ph_edu_dlsu_fx_vision_Contours_nativeDestroyObject
  (JNIEnv *env, jclass type, jlong thiz){
        if (thiz != 0) {
            Contours *self = (Contours *) thiz;
            delete self;
        }
  }


JNIEXPORT void JNICALL Java_ph_edu_dlsu_fx_vision_Contours_apply
  (JNIEnv *env, jclass type, jlong thiz, jlong srcAddr, jlong dstAddr){
      if (thiz != 0) {
          Contours *self = (Contours *) thiz;
          cv::Mat &src = *(cv::Mat *) srcAddr;
          cv::Mat &dst = *(cv::Mat *) dstAddr;
          self->apply(src, dst);
      }
  }



#ifdef __cplusplus
}
#endif
