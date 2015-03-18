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
#include "pixel_similarity.h"
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;
#define CODESEGMENT 1
std::vector<int> cluster_seeds_x;
std::vector<int> cluster_seeds_y;
std::vector<int> selected_cluster_numbers;
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{


    if( event == EVENT_LBUTTONDOWN )
    {
    	cluster_seeds_x.push_back(x);
    	cluster_seeds_y.push_back(y);
    	cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
}
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
      //Mouse selection starts

          Mat img = imread("1.ppm",CV_LOAD_IMAGE_COLOR);
          if ( img.empty() )
                    {
                        cout << "Error loading the image" << endl;
                        return -1;
                    }

          namedWindow( "Display_window", WINDOW_AUTOSIZE );// Create a window for display.
          imshow( "Display_window", img);
          setMouseCallback("Display_window", CallBackFunc, NULL);
          cvWaitKey(0);
          cvDestroyWindow("Display_window");
          printf("\n Out of display window");
          getchar();
      //Mouse selection ends



      // Segmentation starts
      sprintf(filename,"1.ppm");
      image<rgb> *input = loadPPM(filename);
      printf("\nStarting segmentation...\n");
      int *seg = segment_image(input, sigma, k, min_size, &num_ccs);
      //image<rgb> *segmented_image=segment_image_output(input, sigma, k, min_size, &num_ccs);
      //sprintf(filename,"1-segmented.ppm");
      //savePPM(segmented_image, filename);
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
      std::vector <vector <int> > vec_of_clusterwise_plocs;
      std::vector<int> rolling_vec;
      std::vector<int> selected_pixel_indices;

      for(int i=0;i<width*height;i++)
      {
    	  rolling_vec.push_back(indices[i]);
    	  if(i!=width*height-1)
    	  {
    		  if(sorted.at(i)!=sorted.at(i+1))
    		  {
    		  number_of_segments++;
    		  lastpoints.push_back(i);
    		  vec_of_clusterwise_plocs.push_back(rolling_vec);
    		  rolling_vec.clear();
    		  }
    	  }
      }
      if(rolling_vec.size()!=0)
      {
    	  vec_of_clusterwise_plocs.push_back(rolling_vec);
      }

      /*
      printf("\n Number of clusters are %d", vec_of_clusterwise_plocs.size());
      int checksum=0;
      for(int i=0;i<vec_of_clusterwise_plocs.size();i++)
      {
    	  checksum=checksum+vec_of_clusterwise_plocs.at(i).size();
    	  printf("\n The size of cluster %d is %d", i, vec_of_clusterwise_plocs.at(i).size());
      }
      printf("\n Checksum is %d", checksum);
      printf("\n%d number of segments\n", number_of_segments);
      for(int i=0;i<=number_of_segments;i++)
           {
         	  printf("\t Endpoint of segment number %d :%d\n", i,lastpoints.at(i));
           }
      */
      printf("\nStarted search for clusters");
      for(int i=0;i<vec_of_clusterwise_plocs.size();i++)
            {
    	  	  for(int j=0;j<vec_of_clusterwise_plocs.at(i).size();j++)
    	  	  	  {
    	  		  	  for(int k=0;k<cluster_seeds_x.size();k++)
    	  		  	  	  {
    	  		  		  	  if(vec_of_clusterwise_plocs.at(i).at(j)==(cluster_seeds_x.at(k)+width*cluster_seeds_y.at(k)))
    	  		  		  	  selected_cluster_numbers.push_back(i);
    	  		  	  	  }
    	  	  	  }
            }
      for (int k=0;k<selected_cluster_numbers.size();k++)
      {
    	  for (int j=0;j<vec_of_clusterwise_plocs.at(selected_cluster_numbers.at(k)).size();j++)
    	  {
    		  selected_pixel_indices.push_back(vec_of_clusterwise_plocs.at(selected_cluster_numbers.at(k)).at(j));
    	  }
      }
      	  std::vector<size_t> p_indices;
          std::vector<int> p_sorted;
          sort(selected_pixel_indices,p_sorted,p_indices);
          printf("\nTotal number of pixels to move = %d",p_sorted.size());
      getchar();
      // Segmentation ends


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
    printf("Rows: %d Cols: %d\n",video.rows,video.cols);
    //Find the period for each row
    for(row_index=0;row_index<video.rows;row_index++)
    {
    	for(col_index=0;col_index<video.cols;col_index++)
    	{

    		// video.at<cv::Vec3b>(row_index,col_index)[0]; //Gives you Blue value BGR order
    		//printf("\t %d %d sq_difference: %d\n",col_index,i,isequal(video.at<cv::Vec3b>(row_index,col_index)[2],video.at<cv::Vec3b>(row_index,col_index)[1],video.at<cv::Vec3b>(row_index,col_index)[0],video.at<cv::Vec3b>(row_index,i)[2],video.at<cv::Vec3b>(row_index,i)[1],video.at<cv::Vec3b>(row_index,i)[0]));
    		// video.at<cv::Vec3b>(row_index,col_index)[0]; //Gives you Blue value
    	}
    	getchar();
    }

    return 0;
}
