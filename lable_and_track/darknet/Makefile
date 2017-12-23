GPU=1
CUDNN=1
OPENCV=1
DEBUG=0

ARCH= -gencode arch=compute_20,code=[sm_20,sm_21] \
      -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_60,compute_60]

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
LABEL=label_gpu
TRACKING=tracking
OBJDIR=./obj/

CC=icc
NVCC=nvcc --compiler-options '-fPIC'
AR=ar
ARFLAGS=rcs
OPTS=-O3
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wfatal-errors -fPIC

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV -I/work/00410/huang/maverick/opencv3.1.0/include
LDFLAGS+= -L/work/00410/huang/maverick/opencv3.1.0/lib -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_cudaimgproc -lopencv_ml -lopencv_objdetect -lopencv_video -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_tracking -lopencv_cudabgsegm
COMMON+= -DOPENCV -I/work/00410/huang/maverick/opencv3.1.0/include
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/opt/apps/cuda/8.0/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/opt/apps/cuda/8.0/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN
CFLAGS+= -DCUDNN -I/work/00410/huang/maverick/yolo/cudnnv5/include
LDFLAGS+= -L/work/00410/huang/maverick/yolo/cudnnv5/lib64 -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o 
EXECOBJA=captcha.o lsd.o super.o voxel.o art.o tag.o cifar.o go.o rnn.o rnn_vid.o compare.o segmenter.o regressor.o classifier.o coco.o dice.o yolo.o detector.o  writing.o nightmare.o swag.o 
EXECOBJB=label.o
EXECOBJC=tracking.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o network_kernels.o avgpool_layer_kernels.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

all: obj backup results $(SLIB) $(ALIB) $(LABEL) $(TRACKING)


$(LABEL): obj/label.o $(EXECOBJ) $(ALIB)
	icpc -openmp $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(TRACKING): obj/tracking.o $(EXECOBJ) $(ALIB)
	icpc -openmp $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	icc $(COMMON) $(CFLAGS) -c $< -o $@
$(OBJDIR)%.o: %.cpp $(wildcard src/*.h) Makefile include/darknet.hpp
	icpc -openmp $(COMMON) $(CFLAGS) -c $< -o $@


$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(LABEL) $(TRACKING) $(EXECOBJ)

