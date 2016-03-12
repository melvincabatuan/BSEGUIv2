
all:
	g++ -fPIC -o libmhealth_vision.so -shared mHealth_vision.cpp RecolorRC.cpp Contours.cpp ConsensusMatchingTracker.cpp `pkg-config opencv --cflags` `pkg-config opencv --libs` -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -L/home/cobalt/opencv/build2016/lib/ -lopencv_java310
	
clean:
	rm -f *.so
