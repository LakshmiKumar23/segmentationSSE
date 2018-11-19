/*
MIT License
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* This file is generated by nnir2openvx.py on 2018-10-02T11:23:29.742677-07:00 */

#include "annmodule.h"
#include"VX/vx.h"
#include <VX/vx_compatibility.h>
#include <vx_ext_amd.h>
#include <vx_amd_nn.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <string>
#include <inttypes.h>
#include <chrono>
#include <unistd.h>
#include <math.h>
#include <bitset>

#if ENABLE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace cv;
#endif

#include<thread>

#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status     , "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; } }
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return -1; } }

// source: adapted from cityscapes-dataset.org
unsigned char overlayColors[4][20][3] = {
    {
        {200,200,200},      // unclassified
        {128, 64,128},      // road
        {244, 35,232},      // sidewalk
        { 250, 150, 70},    // building
        {102,102,156},      // wall
        {190,153,153},      // fence
        { 0,  0,   0},      // pole
        {250,170, 30},      // traffic light
        {220,220,  0},      // traffic sign
        {0, 255, 0},        // vegetation
        {152,251,152},      // terrain
        { 135,206,250},     // sky
        {220, 20, 60},      // person
        {255,  0,  0},      // rider
        {  0,  0,255},      // car
        {  0,  0, 70},      // truck
        {  0, 60,100},      // bus
        {  0, 80,100},      // train
        {  0,  0,230},      // motorcycle
        {119, 11, 32}       // bicycle
    },
    {
        {225,225,225},      // unclassified
        {160,82,45},        // road
        {0,128,0},          // sidewalk
        {47,79,79},         // building
        {255,240,245},      // wall
        {190,153,153},      // fence
        {240,255,255},      // pole
        {250,170, 30},      // traffic light
        {255,105,180},      // traffic sign
        {124,252,0},        // vegetation
        {75,0,130},         // terrain
        {0,191,255},        // sky
        {230,230,250},      // person
        {230,230,255},      // rider
        {153,50,204},       // car
        {154,60,210},       // truck
        {155,65,215},       // bus
        {107,142,35},       // train
        {128,0,0},          // motorcycle
        {255,255,0}         // bicycle
    },
    {
        {190,190,190},      // unclassified
        {160,82,45},        // road
        {128,0,0},          // sidewalk
        {47,79,79},         // building
        {255,240,245},      // wall
        {190,153,153},      // fence
        {240,255,255},      // pole
        {250,170, 30},      // traffic light
        {255,105,180},      // traffic sign
        {255,255,0},        // vegetation
        {75,0,130},         // terrain
        {0,191,255},        // sky
        {230,230,250},      // person
        {0,0,255},          // rider
        {153,50,204},       // car
        {0,128,128},        // truck
        {0,255,255},        // bus
        {107,142,35},       // train
        {0,128,0},          // motorcycle
        {124,252,0}         // bicycle
    },
    {
        {0,0,0},            // unclassified
        {128, 64,128},      // road
        {244, 35,232},      // sidewalk
        { 250, 150, 70},    // building
        {102,102,156},      // wall
        {190,153,153},      // fence
        {120,120,120},      // pole
        {250,170, 30},      // traffic light
        {220,220,  0},      // traffic sign
        {0, 255, 0},        // vegetation
        {152,251,152},      // terrain
        { 135,206,250},     // sky
        {220, 20, 60},      // person
        {255,  0,  0},      // rider
        {  0,  0,255},      // car
        {  0,  0, 70},      // truck
        {  0, 60,100},      // bus
        {  0, 80,100},      // train
        {  0,  0,230},      // motorcycle
        {119, 11, 32}       // bicycle
    }
};

// source: adapted from cityscapes-dataset.org
std::string segmentationClasses[20] = {
    "Unclassified",
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle"
};

inline int64_t clockCounter()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency()
{
    return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

// probability track bar
const int threshold_slider_max = 100;
int threshold_slider;
double thresholdValue = 0.5;
void threshold_on_trackbar( int, void* ){
    thresholdValue = (double) threshold_slider/threshold_slider_max ;
    return;
}

// alpha track bar
const int alpha_slider_max = 100;
int alpha_slider;
double alphaValue = 0.7;
void alpha_on_trackbar( int, void* ){
    alphaValue = (double) alpha_slider/alpha_slider_max ;
    return;
}

// color track bar
int colorPointer = 0;
const int color_slider_max = 100;
int color_slider;
double colorValue = 0.0;
void color_on_trackbar( int, void* ){
    color_slider = color_slider%10;
    colorValue = (double) color_slider/color_slider_max ;
    if(colorValue <= 0.025) colorPointer = 0;
    else if(colorValue > 0.025 && colorValue <= 0.05) colorPointer = 1;
    else if(colorValue > 0.05 && colorValue <= 0.075) colorPointer = 2;
    else colorPointer = 3;

    // create display legend image
    int fontFace = CV_FONT_HERSHEY_DUPLEX;
    double fontScale = 1;
    int thickness = 1.2;
    cv::Size legendGeometry = cv::Size(325, (20 * 40) + 40);
    Mat legend = Mat::zeros(legendGeometry,CV_8UC3);
    Rect roi = Rect(0,0,325,(20 * 40) + 40);
    legend(roi).setTo(cv::Scalar(255,255,255));
    int l;
    for (l = 0; l < 20; l ++){
        int red, green, blue;
        red = (overlayColors[colorPointer][l][2]) ;
        green = (overlayColors[colorPointer][l][1]) ;
        blue = (overlayColors[colorPointer][l][0]) ;
        std::string className = segmentationClasses[l];
        putText(legend, className, Point(20, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness,8);
        rectangle(legend, Point(225, (l * 40)) , Point(300, (l * 40) + 40), Scalar(red,green,blue),-1);
    }
    cv::imshow("MIVision Image Segmentation", legend);

    return;
}

void findClassProb(size_t start , size_t end, int width, int height, int numClasses, float* output_layer, float threshold, float* prob, unsigned char* classImg)
{
	
	__m128i mask = _mm_set1_epi32(1023);
	for (int i = start; i < end; i+=4)
	{
		
		__m128 vMaxVal = _mm_loadu_ps(&prob[i]);
    	__m128 cur = _mm_loadu_ps(&output_layer[i]); 

		for(int c = 0; c < numClasses; c++)
		{
			vMaxVal = _mm_max_ps(vMaxVal, cur);
		}		
        
        unsigned char out[4]; 
        __m128i vMaxIndex =  _mm_and_si128((__m128i)vMaxVal, mask);
        vMaxIndex = _mm_packus_epi32(vMaxIndex, vMaxIndex);        // Pack down to 16 bits
        vMaxIndex = _mm_packus_epi16(vMaxIndex, vMaxIndex); 
        *(int*)&classImg[i] = _mm_cvtsi128_si32(vMaxIndex); // Store the lower 32 bits
        //*(int*)out = _mm_cvtsi128_si32(vMaxIndex);
        
        /*
        printf("%d\n", out[0]);
        printf("%d\n", out[1]);
        printf("%d\n", out[2]);
        printf("%d\n", out[3]);
        */
        //_mm_stream_si32((int*)&classImg[*(int*)out], 10);
        _mm_storeu_ps(&prob[i], vMaxVal);
		
	}
	
	/*
	__m128i mask = _mm_set1_epi32(1023);//0X000003FF (10 digits for index)
    for(int c = 0; c < numClasses; c++)
    {        
        //__m128i vIndexInc = _mm_set1_epi32(c+1);       
        for(int i = start; i < end; i+=4)
        {
            unsigned char ind[4];
        	//__m128 vMaxVal = _mm_setr_ps(prob[i], prob[i+1], prob[i+2], prob[i+3]);
        	//__m128 cur = _mm_setr_ps(output_layer[i], output_layer[i+1], output_layer[i+2], output_layer[i+3]);   
        	
        	__m128 vMaxVal = _mm_loadu_ps(&prob[i]);
        	__m128 cur = _mm_loadu_ps(&output_layer[i]); 
            //__m128i vMaxIndex = _mm_loadu_si128((__m128i *)&classImg[i]);

           	vMaxVal = _mm_max_ps(vMaxVal, cur);
            __m128i vMaxIndex =  _mm_and_si128((__m128i)vMaxVal, mask);
            vMaxIndex = _mm_packus_epi32(vMaxIndex, vMaxIndex);        // Pack down to 16 bits
            vMaxIndex = _mm_packus_epi16(vMaxIndex, vMaxIndex);        // Pack down to 8 bits (one packing doesn't work for numbers>127)
           	*(int*)ind = _mm_cvtsi128_si32(vMaxIndex); // Store the lower 32 bits
            //classImg[i] = _mm_extract_epi32(vMaxIndex,0); // Store the lower 32 bits
        	_mm_stream_si32((int*)ind, c+1);
            _mm_storeu_ps(&prob[i], vMaxVal);
            //_mm_stream_si32((int*)&classImg[*ind], c+1);

            
            __m128 vcmp = _mm_cmpgt_ps(cur, vMaxVal);
            __m128 vcmp2 = _mm_cmpgt_ps(cur, _mm_set1_ps(threshold));
            vcmp = _mm_and_ps(vcmp, vcmp2);
            vMaxVal = _mm_blendv_ps(vMaxVal, cur, vcmp);
            vMaxIndex = _mm_blendv_epi8(vMaxIndex, vIndexInc, _mm_castps_si128(vcmp));

            _mm_storeu_si128((__m128i *)&classImg[i], vMaxIndex);
            
            

           
        	
            if((output_layer[i] >= threshold) && (output_layer[i] > prob[i]))
            {
                prob[i] = output_layer[i];
                classImg[i] = c + 1;
            }
          	  
          }
       
        output_layer += (width * height);
    }
    */
   
    return;

}

void createMask(size_t start , size_t end, int imageWidth, unsigned char* classImg, Mat& maskImage)
{
    Vec3b pix;
    int classId = 0;
    for(int i = start; i < end; i++)
    {
        for(int j = 0; j < imageWidth; j++)
        {
            classId = classImg[(i * imageWidth) + j];
            pix.val[0] = (overlayColors[colorPointer][classId][2]) ;
            pix.val[1] = (overlayColors[colorPointer][classId][1]) ;
            pix.val[2] = (overlayColors[colorPointer][classId][0]) ;
            maskImage.at<Vec3b>(i, j) = pix;
        }
    }
    return;
}

void getMaskImage(int input_dims[4], float* prob, unsigned char* classImg, float* output_layer, float threshold, cv::Size input_geometry, Mat& maskImage)
{
    int numClasses = input_dims[1];
    int height = input_dims[2];
    int width = input_dims[3];

    int64_t freq = clockFrequency(), t0, t1;
    int numthreads = std::thread::hardware_concurrency();
    //int numthreads = 1;
    //std::cout << "numthreads = " << numthreads << std::endl;
    size_t start = 0, end = 0, chunk = 0;

    // Initialize buffers
    t0 = clockCounter();
    memset(prob, 0, (width * height * sizeof(float)));
    memset(classImg, 0, (width * height));

    t1 = clockCounter();
    //printf("getMaskImage: memset time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    // Class ID generation
    t0 = clockCounter();
    // parallel processing
    start = 0;
    end = height*width;
    chunk = (end - start + (numthreads - 1))/numthreads;
    std::thread t[numthreads] ;
    for(int i = 0 ; i < numthreads ; i++ )
    {
        size_t s = start + i * chunk ;
        size_t e = s + chunk ;
        t[i] = std::thread(findClassProb, s, e, width, height, numClasses, output_layer,threshold, prob, classImg) ;
    }
    for(int i = 0 ; i < numthreads ; i++){ t[i].join() ; }
    t1 = clockCounter();
    printf("getMaskImage: Part 1 time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    // Mask generation
    t0 = clockCounter();
    // parallel create mask
    start = 0;
    end = input_geometry.height;
    chunk = (end - start + (numthreads - 1))/numthreads;
    std::thread M[numthreads] ;
    for(int i = 0 ; i < numthreads ; i++ )
    {
        size_t s = start + i * chunk ;
        size_t e = s + chunk ;
        M[i] = std::thread(createMask, s, e, input_geometry.width, classImg, std::ref(maskImage)) ;
    }
    for(int i = 0 ; i < numthreads ; i++){ M[i].join() ; }
    t1 = clockCounter();
    //printf("getMaskImage: Part 2 time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    return;
}

void processOutput
(
    vx_tensor outputTensor,
    float* output_layer,
    int input_dims[4],
    float* prob,
    unsigned char* classImg,
    cv::Size input_geometry,
    cv::Size output_geometry,
    Mat& inputImage,
    Mat& maskImage
)
{
    // copy output data into local buffer
    int64_t freq = clockFrequency(), t0, t1;
    t0 = clockCounter();
    vx_enum data_type = VX_TYPE_FLOAT32;
    vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];
    vx_map_id map_id;
    float * ptr;
    vx_size count;
    vx_enum usage = VX_READ_ONLY;
    vxQueryTensor(outputTensor, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
    vxQueryTensor(outputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
    vxQueryTensor(outputTensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
    if(data_type != VX_TYPE_FLOAT32) {
        std::cerr << "ERROR: copyTensor() supports only VX_TYPE_FLOAT32: invalid for "  << std::endl;
        return ;
    }
    count = dims[0] * dims[1] * dims[2] * dims[3];
    vx_status status = vxMapTensorPatch(outputTensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for "  << std::endl;
        return ;
    }
    memcpy(output_layer, ptr, (count*sizeof(float)));
    status = vxUnmapTensorPatch(outputTensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for "  << std::endl;
        return ;
    }
    t1 = clockCounter();
    //printf("LIVE: Copy Segmentation Output Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    // process mask img
    t0 = clockCounter();
    float threshold = (float)thresholdValue;
    getMaskImage(input_dims, prob, classImg, output_layer, threshold, input_geometry, maskImage);
    t1 = clockCounter();
    //printf("LIVE: Create Mask Image Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    // Resize and merge outputs img
    t0 = clockCounter();
    //Mat inputDisplay, maskDisplay;
    //cv::resize(inputImage, inputDisplay, cv::Size(output_geometry.width, output_geometry.height));
    //cv::resize(maskImage, maskDisplay, cv::Size(output_geometry.width, output_geometry.height));
    Mat outputDisplay;
    float alpha = alphaValue, beta = ( 1.0 - alpha );
    cv::addWeighted( inputImage, alpha, maskImage, beta, 0.0, outputDisplay);
    t1 = clockCounter();
    //printf("LIVE: Resize and merge Output Image Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    // display img time
    t0 = clockCounter();
    cv::imshow("MIVision Image Segmentation - Input Image", inputImage);
    cv::imshow("MIVision Image Segmentation - Mask Image", maskImage);
    cv::imshow("MIVision Image Segmentation - Merged Image", outputDisplay );
    t1 = clockCounter();
    //printf("LIVE: Output Image Display Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    return;
}

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

int main(int argc, const char ** argv)
{
    // check command-line usage
    if(argc < 2) {
        printf(
            "\n"
            "Usage: MIVisionSeg <weights.bin> --video <video file>\n"
            "\n"
        );
        return -1;
    }
    const char * binaryFilename = argv[1];
    argc -= 2;
    argv += 2;

    std::string videoFile = "../videos/test2.mp4";
    if (argc && !strcasecmp(*argv, "--video"))
    {
        argv++;
        videoFile = *argv;
    }

    // create context, input, output, and graph
    vxRegisterLogCallback(NULL, log_callback, vx_false_e);
    vx_context context = vxCreateContext();
    vx_status status = vxGetStatus((vx_reference)context);
    if(status) {
        printf("ERROR: vxCreateContext() failed\n");
        return -1;
    }
    vxRegisterLogCallback(context, log_callback, vx_false_e);
    vx_graph graph = vxCreateGraph(context);
    status = vxGetStatus((vx_reference)graph);
    if(status) {
        printf("ERROR: vxCreateGraph(...) failed (%d)\n", status);
        return -1;
    }

    // create and initialize input tensor data
    vx_size dims_data[4] = { 2048, 1024, 3, 1 };
    vx_tensor data = vxCreateTensor(context, 4, dims_data, VX_TYPE_FLOAT32, 0);
    if(vxGetStatus((vx_reference)data)) {
        printf("ERROR: vxCreateTensor() failed for data\n");
        return -1;
    }

    // create output tensor loss
    vx_size dims_loss[4] = { 2048, 1024, 19, 1 };
    vx_tensor loss = vxCreateTensor(context, 4, dims_loss, VX_TYPE_FLOAT32, 0);
    if(vxGetStatus((vx_reference)loss)) {
        printf("ERROR: vxCreateTensor() failed for loss\n");
        return -1;
    }

    // build graph using annmodule
    int64_t freq = clockFrequency(), t0, t1;
    t0 = clockCounter();

    status = annAddToGraph(graph, data, loss, binaryFilename);
    if(status) {
        printf("ERROR: annAddToGraph() failed (%d)\n", status);
        return -1;
    }

    status = vxVerifyGraph(graph);
    if(status) {
        printf("ERROR: vxVerifyGraph(...) failed (%d)\n", status);
        return -1;
    }
    t1 = clockCounter();
    printf("OK: graph initialization with annAddToGraph() took %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    t0 = clockCounter();
    status = vxProcessGraph(graph);
    t1 = clockCounter();
    if(status != VX_SUCCESS) {
        printf("ERROR: vxProcessGraph() failed (%d)\n", status);
        return -1;
    }
    printf("OK: vxProcessGraph() took %.3f msec (1st iteration)\n", (float)(t1-t0)*1000.0f/(float)freq);

    t0 = clockCounter();
    int N = 10;
    for(int i = 0; i < N; i++) {
        status = vxProcessGraph(graph);
        if(status != VX_SUCCESS)
            break;
    }
    t1 = clockCounter();
    printf("OK: vxProcessGraph() took %.3f msec (average over %d iterations)\n", (float)(t1-t0)*1000.0f/(float)freq/(float)N, N);

    /***** OPENCV Additions *****/

    // create display windows
    cv::namedWindow("MIVision Image Segmentation");
    cv::namedWindow("MIVision Image Segmentation - Input Image", cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow("MIVision Image Segmentation - Mask Image",cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow("MIVision Image Segmentation - Merged Image",cv::WINDOW_GUI_EXPANDED);

    //create a color track bar
    color_slider = 15;
    cv::createTrackbar("Color Scheme", "MIVision Image Segmentation", &color_slider, color_slider_max, color_on_trackbar);
    //create a probability track bar
    threshold_slider = 50;
    cv::createTrackbar("Probability Threshold", "MIVision Image Segmentation", &threshold_slider, threshold_slider_max, threshold_on_trackbar);
    //create a alpha blend track bar
    alpha_slider = 70;
    cv::createTrackbar("Blend alpha", "MIVision Image Segmentation", &alpha_slider, alpha_slider_max, alpha_on_trackbar);

    // create display legend image
    int fontFace = CV_FONT_HERSHEY_DUPLEX;
    double fontScale = 1;
    int thickness = 1.2;
    cv::Size legendGeometry = cv::Size(325, (20 * 40) + 40);
    Mat legend = Mat::zeros(legendGeometry,CV_8UC3);
    Rect roi = Rect(0,0,325,(20 * 40) + 40);
    legend(roi).setTo(cv::Scalar(255,255,255));
    int l;
    for (l = 0; l < 20; l ++){
        int red, green, blue;
        red = (overlayColors[colorPointer][l][2]) ;
        green = (overlayColors[colorPointer][l][1]) ;
        blue = (overlayColors[colorPointer][l][0]) ;
        std::string className = segmentationClasses[l];
        putText(legend, className, Point(20, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness,8);
        rectangle(legend, Point(225, (l * 40)) , Point(300, (l * 40) + 40), Scalar(red,green,blue),-1);
    }
    cv::imshow("MIVision Image Segmentation", legend);

    cv::Mat frame;
    int total_size = 2048*1024*19*1;
    int input_dims[4]={0};
    input_dims[0] = 1; input_dims[1] = 19;
    input_dims[2] = 1024; input_dims[3] = 2048;
    int outputImgWidth = 1080, outputImgHeight = 720;
    float threshold = 0.01;
    cv::Size input_geometry = cv::Size(input_dims[3], input_dims[2]);
    cv::Size output_geometry = cv::Size(outputImgWidth, outputImgHeight);
    Mat inputDisplay, outputDisplay, maskDisplay;  
    double alpha = 0.8, beta;
    beta = ( 1.0 - alpha );

    // multithread pipeline
    int pipelineDepth = 2;
    std::thread pipeLineThread[pipelineDepth];
    cv::Mat inputFrame[pipelineDepth];
    cv::Mat maskImage[pipelineDepth];
    float *outputBuffer[pipelineDepth];
    unsigned char *classIDBuf[pipelineDepth];
    float *prob[pipelineDepth];
    for(int p = 0; p < pipelineDepth; p++){
        outputBuffer[p] = new float[total_size];
        classIDBuf[p] = new unsigned char[1024 * 2048];
        prob[p] = new float[1024 * 2048];
        maskImage[p].create(input_geometry, CV_8UC3);
    }


    int loopSeg = 1;
    while(argc && loopSeg)
    {
        VideoCapture cap;
        if(!cap.open(videoFile))
            return 0;
        int exitVar = 1;
        int frameCount = 0;
        float msFrame = 0, fpsAvg = 0, frameMsecs = 0;
        int pipelinePointer = -1;
        for(;;)
        {
            // find pipeline pointer number as a variable of pipeline depth
            if((frameCount%pipelineDepth) == 0) pipelinePointer = 0; else pipelinePointer = 1;

            msFrame = 0;
            // capture image frame
            t0 = clockCounter();
            cap >> frame;
            if( frame.empty() ) break; // end of video stream
            t1 = clockCounter();
            msFrame += (float)(t1-t0)*1000.0f/(float)freq;
            //printf("\n\nLIVE: OpenCV Frame Capture Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

            // preprocess image frame
            t0 = clockCounter();
            cv::resize(frame, inputFrame[pipelinePointer], cv::Size(2048,1024));
            t1 = clockCounter();
            msFrame += (float)(t1-t0)*1000.0f/(float)freq;
            //printf("LIVE: OpenCV Frame Resize Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

            // Copy Image frame into the input tensor
            t0 = clockCounter();
            vx_enum usage = VX_WRITE_ONLY;
            vx_enum data_type = VX_TYPE_FLOAT32;
            vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];
            vx_map_id map_id;
            float * ptr;
            vx_size count;

            vxQueryTensor(data, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
            vxQueryTensor(data, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
            vxQueryTensor(data, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
            if(data_type != VX_TYPE_FLOAT32) {
                std::cerr << "ERROR: copyTensor() supports only VX_TYPE_FLOAT32: invalid for " <<  std::endl;
                return -1;
            }
            count = dims[0] * dims[1] * dims[2] * dims[3];
            vx_status status = vxMapTensorPatch(data, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
            if(status) {
                std::cerr << "ERROR: vxMapTensorPatch() failed for " <<  std::endl;
                return -1;
            }
            Mat srcImg;
            for(size_t n = 0; n < dims[3]; n++) {
                srcImg = inputFrame[pipelinePointer];
                for(vx_size y = 0; y < dims[1]; y++) {
                    unsigned char * src = srcImg.data + y*dims[0]*3;
                    float * dstR = ptr + ((n * stride[3] + y * stride[1]) >> 2);
                    float * dstG = dstR + (stride[2] >> 2);
                    float * dstB = dstG + (stride[2] >> 2);
                    for(vx_size x = 0; x < dims[0]; x++, src += 3) {
                        *dstR++ = src[2];
                        *dstG++ = src[1];
                        *dstB++ = src[0];
                    }
                }
            }

            status = vxUnmapTensorPatch(data, map_id);
            if(status) {
                std::cerr << "ERROR: vxUnmapTensorPatch() failed for " <<  std::endl;
                return -1;
            }
            t1 = clockCounter();
            msFrame += (float)(t1-t0)*1000.0f/(float)freq;
            //printf("LIVE: Convert Image to Tensor Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);
   
            // process graph for the input
            t0 = clockCounter();
            status = vxProcessGraph(graph);
            if(status != VX_SUCCESS) break;
            t1 = clockCounter();
            msFrame += (float)(t1-t0)*1000.0f/(float)freq;
            //printf("LIVE: Process Image Segmentation Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

            // launch process output thread
            if((frameCount != 0))
                pipeLineThread[0].join();
            //std::cout << "before  = " << prob[0][1] << std::endl;
            for(int i = 0 ; i < pipelineDepth; i++)
            {
                for(int r = 0; r < input_dims[2]; r++)
                {
                    for(int c = 0; c < input_dims[3]; c++)
                    {
                        union
                        {
                            float input;   // assumes sizeof(float) == sizeof(int)
                            int   output;
                        }   data;

                        //std::cout << r*2048 + c << std::endl; 
                        
                        data.input = prob[i][r*input_dims[3] + c];

                        std::bitset<sizeof(float) * CHAR_BIT>   maxBinary(data.output); 

                        std::bitset<sizeof(float) * CHAR_BIT>   maxIndexBin((i*input_dims[2]+r)*input_dims[3] + c); 
                        std::bitset<32> mask = 0XFFFFFC00;
                        maxBinary = maxBinary & mask;

                        maxBinary = maxBinary | maxIndexBin;
                        //std::cout << maxBinary << std::endl;
                        prob[i][r*input_dims[3] + c] = reinterpret_cast<float &>(maxBinary);
                        //cout << "binary of max after masking= "<< maxBinary << std::endl;
                       
                    }
                }
            }


            //std::cout << "after = " <<prob[0][1] << std::endl;
            pipeLineThread[0] = std::thread(processOutput,
                loss,outputBuffer[pipelinePointer],input_dims,prob[pipelinePointer],
                classIDBuf[pipelinePointer],input_geometry,output_geometry, std::ref(inputFrame[pipelinePointer]),std::ref(maskImage[pipelinePointer]) );
        
            // single output function
            /*
            processOutput(loss,outputBuffer[pipelinePointer],input_dims,prob,
                classIDBuf,input_geometry,output_geometry, inputFrame[pipelinePointer],maskImage[pipelinePointer]);
            */

            // copy output data into local buffer
            /*
            t0 = clockCounter();
            usage = VX_READ_ONLY;
            vxQueryTensor(loss, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
            vxQueryTensor(loss, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
            vxQueryTensor(loss, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
            if(data_type != VX_TYPE_FLOAT32) {
                std::cerr << "ERROR: copyTensor() supports only VX_TYPE_FLOAT32: invalid for "  << std::endl;
                return -1;
            }
            count = dims[0] * dims[1] * dims[2] * dims[3];
            status = vxMapTensorPatch(loss, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
            if(status) {
                std::cerr << "ERROR: vxMapTensorPatch() failed for "  << std::endl;
                return -1;
            }
            memcpy(outputBuffer[pipelinePointer], ptr, (count*sizeof(float)));
            status = vxUnmapTensorPatch(loss, map_id);
            if(status) {
                std::cerr << "ERROR: vxUnmapTensorPatch() failed for "  << std::endl;
                return -1;
            }
            t1 = clockCounter();
            msFrame += (float)(t1-t0)*1000.0f/(float)freq;
            //printf("LIVE: Copy Segmentation Output Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);
            // process mask img
            t0 = clockCounter();
            threshold = (float)thresholdValue;
            getMaskImage(input_dims, prob[pipelinePointer], classIDBuf[pipelinePointer], outputBuffer[pipelinePointer], threshold, input_geometry, maskImage[pipelinePointer]);
            t1 = clockCounter();
            msFrame += (float)(t1-t0)*1000.0f/(float)freq;
            printf("LIVE: Create Mask Image Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);
            // Resize and merge outputs img
            t0 = clockCounter();
            cv::resize(inputFrame[pipelinePointer], inputDisplay, cv::Size(outputImgWidth,outputImgHeight));
            cv::resize(maskImage[pipelinePointer], maskDisplay, cv::Size(outputImgWidth,outputImgHeight));
            alpha = alphaValue; beta = ( 1.0 - alpha );
            cv::addWeighted( inputDisplay, alpha, maskDisplay, beta, 0.0, outputDisplay);
            t1 = clockCounter();
            msFrame += (float)(t1-t0)*1000.0f/(float)freq;
            //printf("LIVE: Resize and merge Output Image Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);
   
            // display img time
            t0 = clockCounter();
            cv::imshow("Input Image", inputDisplay);
            cv::imshow("Mask Image", maskDisplay);
            cv::imshow("Merged Image", outputDisplay );
            t1 = clockCounter();
            msFrame += (float)(t1-t0)*1000.0f/(float)freq;
            //printf("LIVE: Output Image Display Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);
            */
            // calculate FPS
            //printf("LIVE: msec for frame -- %.3f msec\n", (float)msFrame);
            frameMsecs += msFrame;
            if(frameCount && frameCount%10 == 0){
                printf("FPS LIVE: Avg FPS -- %d\n", (int)((ceil)(1000/(frameMsecs/10))));
                frameMsecs = 0;
            }

            // wait to close live inference application
            if( waitKey(2) == 27 ){ pipeLineThread[0].join(); exitVar = 0; loopSeg = 0; break; } // stop capturing by pressing ESC

            frameCount++;
        }
        if(exitVar) pipeLineThread[0].join();

    }

    // release resources
    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseTensor(&data));
    ERROR_CHECK_STATUS(vxReleaseTensor(&loss));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));
    printf("OK: successful\n");

    return 0;
}
