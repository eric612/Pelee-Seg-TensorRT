#include "common.h"
#include "cudaUtility.h"
#include "mathFunctions.h"
#include "pluginImplement.h"
#include "tensorNet.h"
#include "loadImage.h"
#include "imageBuffer.h"
#include <chrono>
#include <thread>

#define BOUND(a,min_val,max_val)           ( (a < min_val) ? min_val : (a >= max_val) ? (max_val) : a 

const char* model  = "model/pelee/pelee_deploy_iplugin.prototxt";
const char* weight = "model/pelee/pelee_merged.caffemodel";

const char* INPUT_BLOB_NAME = "data";

const char* OUTPUT_BLOB_NAME = "detection_out";
static const uint32_t BATCH_SIZE = 1;
volatile bool endvideo = false;
bool csi_cam = false;
//image buffer size = 10
//dropFrame = false
ConsumerProducerQueue<cv::Mat> *imageBuffer = new ConsumerProducerQueue<cv::Mat>(5,csi_cam);

class Timer {
public:
    void tic() {
        start_ticking_ = true;
        start_ = std::chrono::high_resolution_clock::now();
    }
    void toc() {
        if (!start_ticking_)return;
        end_ = std::chrono::high_resolution_clock::now();
        start_ticking_ = false;
        t = std::chrono::duration<double, std::milli>(end_ - start_).count();
        //std::cout << "Time: " << t << " ms" << std::endl;
    }
    double t;
private:
    bool start_ticking_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};


/* *
 * @TODO: unifiedMemory is used here under -> ( cudaMallocManaged )
 * */
float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}


void loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale )
{
    int i;
    int j;
    int line_offset;
    int offset_g;
    int offset_r;
    cv::Mat dst;

    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::resize( input, dst, cv::Size( re_width, re_height ), cv::INTER_LINEAR );
    offset_g = re_width * re_height;
    offset_r = re_width * re_height * 2;
	//#pragma omp parallel for
    for( i = 0; i < re_height; ++i )
    {
        line = dst.ptr< unsigned char >( i );
        line_offset = i * re_width;
        for( j = 0; j < re_width; ++j )
        {
            // b
            unifrom_data[ line_offset + j  ] = (( float )(line[ j * 3 ] - mean.x) * scale);
            // g
            unifrom_data[ offset_g + line_offset + j ] = (( float )(line[ j * 3 + 1 ] - mean.y) * scale);
            // r
            unifrom_data[ offset_r + line_offset + j ] = (( float )(line[ j * 3 + 2 ] - mean.z) * scale);
        }
    }
}
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}
//thread read video
void readPicture()
{
	cv::VideoCapture cap;
	if(csi_cam) {
		int capture_width = 1280 ;
		int capture_height = 720 ;
		int display_width = 1280 ;
		int display_height = 720 ;
		int framerate = 30 ;
		int flip_method = 0 ;

		std::string pipeline = gstreamer_pipeline(capture_width,
		capture_height,
		display_width,
		display_height,
		framerate,
		flip_method);
		std::cout << "Using pipeline: \n\t" << pipeline << "\n";
		cap = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
	}
	else {
		cap = cv::VideoCapture("testVideo/test.avi");
	}
    
    cv::Mat image;
    while(cap.isOpened())
    {
        cap >> image;
        if(image.empty()) {
            endvideo = true;
            break;
        }
        if(!imageBuffer->add(image)) {
			image.release();
		}
    }
}

void MatMul(cv::Mat img1, cv::Mat img2,int r,int g,int b , bool show_mode = false)
{
  int i, j;
  int height = img1.rows;
  int width = img1.cols;
  //LOG(INFO) << width << "," << height << "," << img2.rows << "," << img2.cols;
  //#pragma omp parallel for

  for (i = 0; i < height; i++) {
    unsigned char* ptr1 = img1.ptr<unsigned char>(i);
    const unsigned char* ptr2 = img2.ptr<unsigned char>(i);
    int img_index1 = 0;
    int img_index2 = 0;
    for (j = 0; j < width; j++) {
      if(ptr2[img_index2]>90) {
        if(show_mode) {
          ptr1[img_index1] = b;
          ptr1[img_index1+1] = g;
          ptr1[img_index1+2] = r;
        }
        else {
          ptr1[img_index1] = b/2 + ptr1[img_index1]/2;
          ptr1[img_index1+1] = g/2 + ptr1[img_index1]/2;
          ptr1[img_index1+2] = r/2 + ptr1[img_index1]/2;
        }

      }
      //ptr1[img_index1+idx] = (unsigned char) BOUND(ptr1[img_index1] + ptr2[img_index2] * 1.0,0,255);
      //ptr1[img_index1+1] = (ptr2[img_index2]);
      //ptr1[img_index1+2] = (unsigned char) BOUND(ptr1[img_index1+2] + (255-ptr2[img_index2]) * 0.4,0,255);
      //ptr1[img_index1+2] = (unsigned char) BOUND((ptr2[img_index2]) ,0,255);
      img_index1+=3;
      img_index2++;
    }
  }

}
int main(int argc, char *argv[])
{
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME,"sigmoid"};
    TensorNet tensorNet;
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,BATCH_SIZE);

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);
	DimsCHW dimsOut2  = tensorNet.getTensorDims("sigmoid");
    float* data    = allocateMemory( dimsData , (char*)"input blob");
    std::cout << "allocate data" << std::endl;
    float* output  = allocateMemory( dimsOut  , (char*)"output blob");
    std::cout << "allocate output" << std::endl;
	float* output2  = allocateMemory( dimsOut2  , (char*)"output blob 2");
    std::cout << "allocate output2" << std::endl;
    int height = 304;
    int width  = 304;
	void* imgCPU;
    void* imgCUDA;
	const size_t size = width * height * sizeof(float3);

	if( CUDA_FAILED( cudaMalloc( &imgCUDA, size)) )
	{
		cout <<"Cuda Memory allocation error occured."<<endl;
		return false;
	}
    cv::Mat frame,srcImg;

	
    Timer timer;
    std::thread readTread(readPicture);
    readTread.detach();
    double msTime_avg = 0.;
    int count = 0;
    int ch_size = dimsOut2.c();
	std::vector<cv::Mat> seg_img;
	for(int i = 0; i<ch_size;i++) {   
		seg_img.push_back(cv::Mat(76, 76, CV_8UC1));
	}
	std::vector<int> color = {128,255,128,244,35,232};
    while(1){
		if(endvideo && imageBuffer->isEmpty()) {
			break;
		}
		imageBuffer->consume(frame);

		if(!frame.rows) {
			break;
		}
		//srcImg = frame.clone();
		cv::resize(frame, srcImg, cv::Size(304,304));


		void* imgData = malloc(size);
		//memset(imgData,0,size);
		
		loadImg(srcImg,height,width,(float*)imgData,make_float3(103.94,116.78,123.68),0.017);
		
		cudaMemcpyAsync(imgCUDA,imgData,size,cudaMemcpyHostToDevice);
		
		void* buffers[] = { imgCUDA, output , output2}; 

		

		timer.tic();
		tensorNet.imageInference( buffers, output_vector.size() + 1, BATCH_SIZE);
		timer.toc();
		double msTime = timer.t;
		
		msTime_avg+= msTime;
		count++;
		std::cout<<msTime_avg/(float)count<< std::endl;	
		vector<vector<float> > detections;

		for (int k=0; k<100; k++)
		{
			if(output[7*k+1] == -1)
				break;
			float classIndex = output[7*k+1];
			float confidence = output[7*k+2];
			float xmin = output[7*k + 3];
			float ymin = output[7*k + 4];
			float xmax = output[7*k + 5];
			float ymax = output[7*k + 6];
			//std::cout << classIndex << " , " << confidence << " , "  << xmin << " , " << ymin<< " , " << xmax<< " , " << ymax << std::endl;
			int x1 = static_cast<int>(xmin * frame.cols);
			int y1 = static_cast<int>(ymin * frame.rows);
			int x2 = static_cast<int>(xmax * frame.cols);
			int y2 = static_cast<int>(ymax * frame.rows);
			cv::rectangle(frame,cv::Rect2f(cv::Point(x1,y1),cv::Point(x2,y2)),cv::Scalar(255,0,255),1);

		}
		int scale = 4;

		int w = width / scale;
		int h = height / scale;
		
		for(int c = 0; c<seg_img.size();c++) { 
			int img_index1 = 0;  
			for (int y = 0; y < h; y++) {
				uchar* ptr2 = seg_img[c].ptr<uchar>(y);
				int img_index2 = 0;
				for (int j = 0; j < w; j++) {
					int val = output2[img_index1+c*w*h] * 255;
					if (val>255) val = 255; 
					if (val<0) val = 0;
					ptr2[img_index2] = (unsigned char)val;
					//if(c==1)
					//  printf("%f\n",result2[img_index1+c*w*h]);
					img_index1++;
					img_index2++;
				}
			}
		}
		cv::Mat seg_img_resized;
		for(int i=0;i<seg_img.size();i++) {
          cv::resize(seg_img[i], seg_img_resized, cv::Size(frame.cols, frame.rows),cv::INTER_AREA);
          int color_index = (i)*3;
          MatMul(frame, seg_img_resized,color[color_index],color[color_index+1],color[color_index+2]);
        }
		//cv::namedWindow("show", cv::WINDOW_NORMAL);
		//cv::resizeWindow("show", 400, 400);
		cv::imshow("show", frame);
		cv::waitKey(1);
		free(imgData);
		frame.release();
		srcImg.release();
    }
    cudaFree(imgCUDA);
    cudaFreeHost(imgCPU);
    cudaFree(output);
    tensorNet.destroy();
    return 0;
}
