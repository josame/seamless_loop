//============================================================================
// Name        : seamless_loop.cpp
// Author      : ameya
// Version     :
// Copyright   : Your copyright notice
// Description : Seamless loop in C++, Ansi-style
//============================================================================
#include "opencv2/opencv.hpp"
#include <string>
#include <sstream>
#include<stdio.h>
#include<vector>
using namespace cv;
#define CODESEGMENT 1
int main(int, char**)
{
	int tot_num_frames_in_vid=0;
	//char * filename = new char[100];
    Mat frame;
    cv::Mat all_pixel_Mat;
    Mat video;
    int flag=0;
    int row_index,col_index;
    VideoCapture cap("SamplePoolPalms.mp4");
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    while(flag==0)
    {
        cap >> frame; // get a new frame from video
        if(frame.rows>0 && frame.cols>0)
        {
        tot_num_frames_in_vid=tot_num_frames_in_vid+1;
        frame=frame.reshape(0,1);
        video.push_back(frame);
        }
        else
        {
        	flag=1;
        	cap.release();
        }
    }
    transpose(video,video);
    //Find the period for each row
    for(row_index=0;row_index<video.rows;row_index++)
    {
    	for(col_index=0;col_index<video.cols;col_index++)
    	{
    		video.at<cv::Vec3b>(row_index,col_index)[0]; //Blue
    	}
    }

    return 0;
}
