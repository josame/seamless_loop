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
#include <cstdio>
#include <cstdlib>
#include <image.h>
#include <misc.h>
#include <pnmfile.h>
#include "segment-image2.h"
#include "sort.h"
using namespace cv;
using namespace std;
#define CODESEGMENT 1
int main(int, char**)
{
	int tot_num_frames_in_vid=0;
	char * filename = new char[100];
    Mat frame;
    cv::Mat all_pixel_Mat;
    Mat video;
    int flag=0;
    int row_index,col_index;
    float sigma = 0.8;
      float k = 300;
      int min_size = 1000;
      int num_ccs;

      //Working code for graph segmentation

      // Segmentation starts
      sprintf(filename,"1.ppm");
      image<rgb> *input = loadPPM(filename);
      int *seg = segment_image(input, sigma, k, min_size, &num_ccs);
      //sprintf(filename,"1-segmented.ppm");
      //savePPM(seg, filename);
      printf("\nDone segmenting\n");
      int width=input->width();
      int height=input->height();
      // Sorting start
      std::vector<int> a(seg, seg + width*height);
      /*for(int i=100;i>1;i--)
            {
          	  printf("\t %d", a.at(height*width-i-1));
            }*/
      std::vector<size_t> indices;
      std::vector<int> sorted;

      sort(a,sorted,indices);
      /*
      for(int i=5000;i<6000;i++)
      {
    	  printf("\t Cluster number:%d Pixel: (%d,%d) \n", sorted.at(i),indices[i]%width, indices[i]/width);
      }*/
      int number_of_segments=0;
      std::vector<int> lastpoints;
      for(int i=0;i<width*height-1;i++)
      {
    	  if(sorted.at(i)!=sorted.at(i+1))
    	  {
    		  number_of_segments++;
    		  lastpoints.push_back(i);
    	  }
      }
      lastpoints.push_back(width*height-1);
      printf("\n%d number of segments\n", number_of_segments);
      /*for(int i=0;i<=number_of_segments;i++)
           {
         	  printf("\t Endpoint of segment number %d :%d\n", i,lastpoints.at(i));
           }*/
      //Cutting the points by segments

      // Segmentation ends
      getchar();


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
    		// video.at<cv::Vec3b>(row_index,col_index)[0]; //Gives you Blue value
    	}
    }

    return 0;
}
