#include "common.h"
#include "cudaUtility.h"
#include "mathFunctions.h"
#include "pluginImplement.h"
#include "tensorNet.h"
#include "loadImage.h"
#include "imageBuffer.h"
#include <chrono>
#include <thread>


const char* model  = "model/pelee/pelee_deploy_iplugin.prototxt";
const char* weight = "model/pelee/pelee_merged.caffemodel";

const char* INPUT_BLOB_NAME = "data";

const char* OUTPUT_BLOB_NAME = "detection_out";
static const uint32_t BATCH_SIZE = 1;
volatile bool endvideo = false;
//image buffer size = 10
//dropFrame = false
ConsumerProducerQueue<cv::Mat> *imageBuffer = new ConsumerProducerQueue<cv::Mat>(5,true);

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
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    //cv::VideoCapture cap("testVideo/test.avi");
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

int main(int argc, char *argv[])
{
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME};
    TensorNet tensorNet;
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,BATCH_SIZE);

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);

    float* data    = allocateMemory( dimsData , (char*)"input blob");
    std::cout << "allocate data" << std::endl;
    float* output  = allocateMemory( dimsOut  , (char*)"output blob");
    std::cout << "allocate output" << std::endl;
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

		void* buffers[] = { imgCUDA, output }; 

		

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
		cv::imshow("Pelee",frame);
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
