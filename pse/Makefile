CXXFLAGS = -I include  -std=c++11 -O3 -I /home/zhm/anaconda3/envs/mmdetection/include

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp
#OPENCV = `pkg-config --cflags --libs opencv`
OPENCV=/home/zhm/anaconda3/envs/mmdetection/lib/
LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)