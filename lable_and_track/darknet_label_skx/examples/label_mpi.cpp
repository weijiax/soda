#include "darknet.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
//#include <opencv2/tracking.hpp>
//#include <opencv2/cudabgsegm.hpp>

#include <iostream>

using namespace cv;
using namespace std;

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <pthread.h>
#include <mpi.h>

#define MAX_OBJ	(512)
#define MAX_THREAD	(64)

typedef struct	{
	int x, y, xMax, yMax;
	int type;	// 0 - people;	1 - car or truck;  2- bus
	int prob;
}OBJECT,*POBJECT;


image *im;

//Mat frame; //current frame
Mat *frame_ListA; 
Mat *frame_ListB;
Mat *pFrame_label, *pFrame_read, *pframe_List[2];	// the pointer either pointing to list A or B
int *tid;
int IdxRead=0;

FILE *fOut[MAX_THREAD];

double fps, fps_inv, t_video;
int width, height;
int nFrame=0, nFrame_Done=0;

int Finished=0;
int nWorker=4;

int nnode;
int node_idx;
//int worker_id;	// nworker * node_idx + tid
int FrameTotal, FrameToWork, FrameToStart;

image **alphabet;
network *net;
layer *last_layer;
char **names;

void Mat_into_image(Mat src, image im);
void *func_read_video( void *ptr );	// the thread for reading video into buffer
void *label_image( void *ptr );		// the thread to call yolo to label image
void Merge_Output(void);
void *init_worker( void *ptr );

void *label_image( void *ptr )		// the thread to call yolo to label image
{
	int tid;
	int i, j, whn;
	int nObj_Yolo=0;
	OBJECT Yolo_ObjList[MAX_OBJ];
    float nms=.4;
    float thresh=0.24;
    float hier_thresh=0.50;

	tid = *((int*)ptr);

	if(pFrame_label[tid].empty())	{
		return 0;
	}
	
	Mat_into_image(pFrame_label[tid], im[tid]);
	image sized = letterbox_image(im[tid], net[tid].w, net[tid].h);
    
	whn = last_layer[tid].w*last_layer[tid].h*last_layer[tid].n;
	box *boxes = (box *)calloc(whn, sizeof(box));
	float **probs = (float **)calloc(whn, sizeof(float *));
	for(j = 0; j < whn; ++j) probs[j] = (float *)calloc(last_layer[tid].classes + 1, sizeof(float *));
    
	float *X = sized.data;
	network_predict(net[tid], X);
	get_region_boxes(last_layer[tid], im[tid].w, im[tid].h, net[tid].w, net[tid].h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
	if (nms) do_nms_obj(boxes, probs, whn, last_layer[tid].classes, nms);
    
	draw_detections_Ex(im[tid], whn, thresh, boxes, probs, names, alphabet, last_layer[tid].classes, &nObj_Yolo, (int*)Yolo_ObjList);
	
	fprintf(fOut[tid], "%5d %3d\n", FrameToStart + nFrame_Done+tid+1, nObj_Yolo);
	for(i=0; i<nObj_Yolo; i++) {
		fprintf(fOut[tid], "%d %d %d %d %d %d\n", Yolo_ObjList[i].type, Yolo_ObjList[i].x, Yolo_ObjList[i].y, Yolo_ObjList[i].xMax, Yolo_ObjList[i].yMax, Yolo_ObjList[i].prob);
	}
	
	free_image(sized);
	free(boxes);
	free_ptrs((void **)probs, whn);

	if(node_idx == 0)	printf("Thread %d: frame = %d\n", tid, FrameToStart + nFrame_Done+tid+1);

	return 0;
}

void *init_worker( void *ptr )
{
	int tid;
	char szName[256];

	tid = *((int*)ptr);
	
	net[tid] = parse_network_cfg("cfg/yolo.cfg");

	sprintf(szName, "/tmp/yolo.weights_%d", tid%3);	// three files for reading
	load_weights(&(net[tid]), szName);
	
	set_batch_network(&(net[tid]), 1);
	last_layer[tid] = net[tid].layers[net[tid].n-1];

	return 0;
}


int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nnode);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_idx);

    pthread_t thread_read, *thread_worker;
    static struct timeval tm1, tm2;
    list *options = read_data_cfg("cfg/coco.data");
    char *name_list = option_find_str(options, "names", "data/names.list");
    names = get_labels(name_list);
    
	clock_t time;
    char buff[256], szTxt[128];
	char szOutName[256];
    int i,j;
    int nRealWorker;
    int iret;

    gpu_index = -1;

	nWorker = atoi(argv[2]);

    alphabet = load_alphabet();
	tid = (int *)malloc(sizeof(int)*nWorker);
	net = (network *)malloc(sizeof(network)*nWorker);
	last_layer = (layer *)malloc(sizeof(layer)*nWorker);
	im = (image *)malloc(sizeof(image)*nWorker);

	thread_worker = (pthread_t *)malloc(sizeof(pthread_t)*nWorker);;

	for(i=0; i<nWorker; i++)	{
		tid[i] = i;
	}
	
	for(i=0; i<nWorker; i++)	{
		iret = pthread_create( &(thread_worker[i]), NULL, init_worker, (void*)(&(tid[i])));
		if(iret)	{
			fprintf(stderr,"Error - pthread_create() init_worker return code: %d\n",iret);
			exit(EXIT_FAILURE);
		}
	}
	
	for(i=0; i<nWorker; i++)	{
		pthread_join( thread_worker[i], NULL);
	}


    srand(2222222);

    VideoCapture capture(argv[1]);

    if(!capture.isOpened()){
        cerr << "Unable to open video file: " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }
    width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    fps = capture.get(CV_CAP_PROP_FPS);
    fps_inv = 1.0/fps;
	FrameTotal = (int)(capture.get(CV_CAP_PROP_FRAME_COUNT) + 0.001);

	int Frame_per_Node;
	if(FrameTotal%nnode == 0)	{
		Frame_per_Node = FrameTotal/nnode;
		FrameToStart = Frame_per_Node*node_idx;
		FrameToWork = Frame_per_Node;
	}
	else	{
		Frame_per_Node = FrameTotal/nnode + 1;
		FrameToStart = Frame_per_Node*node_idx;
		if( node_idx == (nnode-1) )	{	// the last task
			FrameToWork = max(FrameTotal - Frame_per_Node*(nnode-1), 0 );
		}
		else	{
			FrameToWork = Frame_per_Node;
		}
	}
    
	for(i=0; i<nWorker; i++)	{
		im[i] = make_image(width, height, 3);	// color image. three channels
	}
    
	for(i=0; i<nWorker; i++)	{
		sprintf(szOutName, "yolo_%d.log", nWorker * node_idx + i);	// worker id
	    fOut[i] = fopen(szOutName, "w");
	}

	frame_ListA = new Mat[nWorker];
	frame_ListB = new Mat[nWorker];
	pframe_List[0] = frame_ListA;
	pframe_List[1] = frame_ListB;

	pFrame_read = pframe_List[IdxRead];

	capture.set(CV_CAP_PROP_POS_FRAMES, (double)FrameToStart);	// starting from this frame
	func_read_video(&capture);
	
	IdxRead = (IdxRead+1)%2;
	pFrame_read = pframe_List[IdxRead];
	pFrame_label = pframe_List[(IdxRead+1)%2];

	MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&tm1, NULL);

    while(1)	{
		if(! Finished)	{
			iret = pthread_create( &thread_read, NULL, func_read_video, (void*)(&capture));
			if(iret)	{
				fprintf(stderr,"Error - pthread_create() func_read_video return code: %d\n",iret);
				exit(EXIT_FAILURE);
			}
		}

		nRealWorker = 0;
		for(i=0; i<nWorker; i++)	{
			if( ! pFrame_label[i].empty() )	{
				iret = pthread_create( &(thread_worker[i]), NULL, label_image, (void*)(&(tid[i])));
				if(iret)	{
					fprintf(stderr,"Error - pthread_create() label_image return code: %d\n",iret);
					exit(EXIT_FAILURE);
				}
				nRealWorker++;
			}
		}

		if(! Finished)	{
			pthread_join( thread_read, NULL);
		}
		
		for(i=nWorker-1; i>=0; i--)	{
			if( ! pFrame_label[i].empty() )	{
				pthread_join( thread_worker[i], NULL); 
			}
		}
		if(nRealWorker < nWorker)	{	// all jobs are done
			break;
		}

		IdxRead = (IdxRead+1)%2;
		pFrame_read = pframe_List[IdxRead];
		pFrame_label = pframe_List[(IdxRead+1)%2];
		if(Finished)	pFrame_read = NULL;
		nFrame_Done += nRealWorker;

		if(nFrame_Done >= FrameToWork)	{
			break;
		}
    }

    capture.release();
	for(i=0; i<nWorker; i++)	{
	    fclose(fOut[i]);
	    free_image(im[i]);
		if(! frame_ListA[i].empty())	{
			frame_ListA[i].~Mat();
		}
		if(! frame_ListB[i].empty())	{
			frame_ListB[i].~Mat();
		}
	}

	free(tid);
	free(net);
	free(last_layer);
	free(thread_worker);

	delete[] frame_ListA;
	delete[] frame_ListB;

	MPI_Barrier(MPI_COMM_WORLD);

	if(node_idx == 0)	{
		printf("nFrame = %d\n", FrameTotal);
		gettimeofday(&tm2, NULL);

		unsigned long long t = 1000 * (tm2.tv_sec - tm1.tv_sec) + (tm2.tv_usec - tm1.tv_usec) / 1000;
		printf("%llu ms. %4.1lf fps\n", t, (double)(FrameTotal/( (double)(0.001*(double)(t))  )) );


		Merge_Output();
	}

    MPI_Finalize();

    return 0;
}

void *func_read_video( void *ptr )
{
	int i;
	VideoCapture *pcapture;

	pcapture = (VideoCapture *)ptr;

	for(i=0; i<nWorker; i++)	{	// release existing image
		if(! pFrame_read[i].empty())	{
			pFrame_read[i].~Mat();
		}
	}

	if(Finished)	{
		return;
	}

	i = 0;
	while(i<nWorker)	{
		if(! pcapture->read(pFrame_read[i])) {
			Finished = 1;	// no more data available or error in reading file
			break;
		}
		else	{
			i++;
			if( (nFrame+i) == FrameToWork )	{	// have finished all jobs scheduled
				Finished = 1;
				break;
			}
		}
	}
	nFrame += i;

	return 0;
}



void Mat_into_image(Mat src, image im)
{    
    unsigned char *data = (unsigned char *)(src.data);
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    int step = src.step;
    int i, j, k, w_h, w_h_2;

    w_h = w * h;
    w_h_2 = w_h*2;

    for(i = 0; i < h; ++i){
        for(j = 0; j < w; ++j){
            for(k= 0; k < c; ++k){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;	// convert to float
            }
        }
    }

    k = 0;	// Red
    for(i = 0; i < h; ++i){
        for(j = 0; j < w; ++j){
             im.data[i*w + j] = data[i*step + j*c + 2]/255.;	// convert to float
        }
    }

    k = 1;	// Green
    for(i = 0; i < h; ++i){
        for(j = 0; j < w; ++j){
             im.data[w_h + i*w + j] = data[i*step + j*c + 1]/255.;	// convert to float
        }
    }

    k = 2;	// Blue
    for(i = 0; i < h; ++i){
        for(j = 0; j < w; ++j){
             im.data[w_h_2 + i*w + j] = data[i*step + j*c    ]/255.;	// convert to float
        }
    }
//    rgbgr_image(out);	// swap R<->B
}

void Merge_Output(void)
{
	FILE *fIn[MAX_THREAD*64], *fOut;	// max_node = 64
	char szName[256], szLine[256], *ReadLine;
	int i, j, idx, tid, nReadItem, Frame_Yolo, nObj_Yolo, nThreads, idx_node, worker_id;

	nThreads = nWorker*nnode;
	for(i=0; i<nThreads; i++)	{
		sprintf(szName, "yolo_%d.log", i);
		fIn[i] = fopen(szName, "r");
		if(fIn[i] == NULL)	{
			printf("Fail to open file: %s\nQuit.\n", szName);
			exit(1);
		}
	}

	fOut = fopen("yolo.log", "w");
	
	for(idx=1; idx<=FrameTotal; idx++)	{
		idx_node = (idx-1) / FrameToWork;	// the index of node
		tid = ( (idx-1) % FrameToWork ) % nWorker;
		worker_id = nWorker * idx_node + tid;
		
		ReadLine = fgets(szLine, 256, fIn[worker_id]);
		if( (ReadLine == NULL) || feof(fIn[worker_id]) )	{
			printf("Error in reading Yolo detection record. tid = %d\nExit\n", worker_id);
			fclose(fIn[worker_id]);
			exit(1);
		}
		
		nReadItem = sscanf(szLine, "%d%d", &Frame_Yolo, &nObj_Yolo);
		if(nReadItem != 2)	{
			printf("Error in reading Yolo detection record. tid = %d\nExit\n", worker_id);
			fclose(fIn[worker_id]);
			exit(1);
		}
		if(Frame_Yolo != idx)	{
			printf("Error in reading Yolo detection record.\nFrame_Yolo != idx   tid = %d\n", worker_id);
			fclose(fIn[worker_id]);
			exit(1);
		}
		
		fprintf(fOut, "%s", szLine);
		for(i=0; i<nObj_Yolo; i++)	{
			ReadLine = fgets(szLine, 256, fIn[worker_id]);
			if( (ReadLine == NULL) || feof(fIn[worker_id]) )	{
				printf("Error in reading Yolo detection record. tid = %d\nExit\n", worker_id);
				fclose(fIn[worker_id]);
				exit(1);
			}
			fprintf(fOut, "%s", szLine);
		}
	}
	
	
	for(i=0; i<nThreads; i++)	{
		fclose(fIn[i]);
	}
	fclose(fOut);

}

