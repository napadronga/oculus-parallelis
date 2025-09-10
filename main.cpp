#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>

using namespace std;
using namespace cv;

VideoCapture cap(0);

int main(int argc, char** argv) {

    cuda::printCudaDeviceInfo(0);

    Mat img;
    cuda::GpuMat imgGpu;


    while (cap.isOpened()) {
        auto start = getTickCount();

        //first, read a frame from the camera to cpu memory
        cap.read(img);

        //then, upload the frame from cpu to gpu memory
        imgGpu.upload(img);

        //gpu processing
        cuda::cvtColor(imgGpu, imgGpu, COLOR_BGR2GRAY);

        auto gaussianFilter = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, {3, 3}, 1);
        gaussianFilter->apply(imgGpu, imgGpu);

        //download back to cpu for display
        imgGpu.download(img);

        auto end = getTickCount();
        auto totalTime = (end - start) / getTickFrequency();
        auto fps = 1.0 / totalTime;

        putText(img, "FPS: " + to_string(int(fps)), Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 2);
        imshow("Image", img);

        if (waitKey(1) == 27){
            break;
        }
    }

    return 0;
}

