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
#define NO_OF_FRAMES 200
Mat input_video[NO_OF_FRAMES];
Mat input_video_bgr[NO_OF_FRAMES];
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
class xPixel {
public:
	unsigned char period;
	unsigned char start;
	unsigned char pixel_values[NO_OF_FRAMES];
	float pixel_values_variance_arr[NO_OF_FRAMES];
	float pixel_values_temporal_variance;
	unsigned char is_static;
	unsigned char new_period;
	unsigned char stage1_s_arr[NO_OF_FRAMES];
	float stage1_e_arr[NO_OF_FRAMES];
	unsigned int mad_vals[3][3];
	xPixel() {
		period = 32;
		start = 0;
		pixel_values_temporal_variance = 1000;
		is_static = 0;
		for (int i = 0; i < NO_OF_FRAMES; i++) {
			pixel_values[i] = 0;
			pixel_values_variance_arr[i] = 0;
			stage1_s_arr[i] = 0;
			stage1_e_arr[i] = 0;
		}
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				mad_vals[i][j] = 0;
			}
		}
	}
	void calc_variance (unsigned int num_of_frames, unsigned int num_of_cols) {
		for (unsigned int i = 0; i < num_of_frames-1; i++) {
			pixel_values_variance_arr[i]=abs((float)pixel_values[i]-(float)pixel_values[i+1]);
		}
		for (unsigned int i = 0; i < num_of_frames-1; i++) {
			for (unsigned int j = i+1; j < num_of_frames; j++) {
				float ival = pixel_values_variance_arr[j];
				float jval = pixel_values_variance_arr[j];
				if (ival > jval) {
					pixel_values_variance_arr[j] = ival;
					pixel_values_variance_arr[i] = jval;
				}
			}
		}
		int min_val = 100000;
		int max_val = -1;
		for (unsigned int i = 0; i < num_of_frames; i++) {
			if (pixel_values[i] < min_val) min_val = pixel_values[i];
			if (pixel_values[i] > max_val) max_val = pixel_values[i];
		}
		pixel_values_temporal_variance = pixel_values_variance_arr[num_of_frames / 2];
		if ((max_val - min_val) < 10) {
			//cout << (max_val - min_val);
			is_static = 1;
			period = 1;
		}

		xPixel * pixel = this;
		xPixel * s_arr[3][3];
		s_arr[0][0] = (pixel - num_of_cols - 1);
		s_arr[0][1] = (pixel - num_of_cols - 0);
		s_arr[0][2] = (pixel - num_of_cols + 1);
		s_arr[1][0] = (pixel - 0 - 1);
		s_arr[1][1] = (pixel - 0 - 0);
		s_arr[1][2] = (pixel - 0 + 1);
		s_arr[2][0] = (pixel + num_of_cols - 1);
		s_arr[2][1] = (pixel + num_of_cols - 0);
		s_arr[2][2] = (pixel + num_of_cols + 1);
		unsigned int mad[NO_OF_FRAMES];
		for (unsigned int i = 0; i < 3; i++) {
			for (unsigned int j = 0; j < 3; j++) {
				for (unsigned int t = 0; t < num_of_frames; t++) {
					mad[t] = abs(s_arr[1][1]->pixel_values[t] - s_arr[i][j]->pixel_values[t]);
				}
				for (unsigned int p = 0; p < num_of_frames - 1; p++) {
					for (unsigned int q = p + 1; q < num_of_frames; q++) {
						unsigned int  ival = mad[p];
						unsigned int  jval = mad[q];
						if (ival > jval) {
							mad[q] = ival;
							mad[p] = jval;
						}
					}
				}
				mad_vals[i][j] = mad[int(num_of_frames / 2)];
			}
		}
	}
};

unsigned int phi(unsigned int start, unsigned int period, unsigned int t) {
	unsigned int retval = start + period - (start % period) + (t % period);
	if (retval >= start + period) {
		retval -= period;
	}
	//printf("%d %d %d %d\n", start, period, t, retval);
	return retval;
	//return (start + abs(((signed)t - (signed)start)) % (signed)period));
}

int gcd(int a, int b)
{
	for (;;) {
		if (a == 0) return b;
		b %= a;
		if (b == 0) return a;
		a %= b;
	}
}

int lcm(int a, int b)
{
	int temp = gcd(a, b);
	return temp ? (a / temp * b) : 0;
}

float get_pixel_etemporal(xPixel * pixel, unsigned int num_of_frames, unsigned int s, unsigned int p) {

	if (pixel->is_static) return 0.0;

	if ((s + p) > num_of_frames) {
		printf("ERROR: S+P > NUM_FRAMES");
		return 10000;
	}
	unsigned int s_plus_p = pixel->pixel_values[s+p];
	int diff = s - s_plus_p;
	unsigned int lambda = 400;
	float gamma = 1 / (1 + lambda*pixel->pixel_values_temporal_variance);
	return (diff * diff) * gamma;
}

float get_pixel_espatial_stage1(xPixel * pixel, unsigned int num_of_frames, unsigned int num_of_cols, unsigned int sx, unsigned int p) {

	xPixel * s_arr[3][3];
	s_arr[0][0] = (pixel - num_of_cols - 1);
	s_arr[0][1] = (pixel - num_of_cols - 0);
	s_arr[0][2] = (pixel - num_of_cols + 1);
	s_arr[1][0] = (pixel - 0 - 1);
	s_arr[1][1] = (pixel - 0 - 0);
	s_arr[1][2] = (pixel - 0 + 1);
	s_arr[2][0] = (pixel + num_of_cols - 1);
	s_arr[2][1] = (pixel + num_of_cols - 0);
	s_arr[2][2] = (pixel + num_of_cols + 1);

	float V = 0;
	for (unsigned int i = 0; i < 3; i++) {
		for (unsigned int j = 0; j < 3; j++) {
			if (i==1 && j==1) continue;
			float Vxz = 0.0;
			for (unsigned int t = 0; t < p; t++) {
				//unsigned int sx = s_arr[1][1]->stage1_s_arr[p];
				unsigned int sz = s_arr[i][j]->stage1_s_arr[p];
				int Vx = (s_arr[1][1]->pixel_values[phi(sx, p, t)] - s_arr[1][1]->pixel_values[phi(sz, p, t)]);
				int Vz = (s_arr[i][j]->pixel_values[phi(sx, p, t)] - s_arr[i][j]->pixel_values[phi(sz, p, t)]);
				Vxz += (Vx*Vx) + (Vz*Vz);
			}
			Vxz /= p;
			float gammaxz = (1 / (1 + (100 * (float)s_arr[1][1]->mad_vals[i][j])));
			V += (Vxz * gammaxz);
		}
	}

	return V;
}

float get_pixel_espatial_stage2(xPixel * pixel, unsigned int num_of_frames, unsigned int num_of_cols, unsigned int p) {

	xPixel * s_arr[3][3];
	s_arr[0][0] = (pixel - num_of_cols - 1);
	s_arr[0][1] = (pixel - num_of_cols - 0);
	s_arr[0][2] = (pixel - num_of_cols + 1);
	s_arr[1][0] = (pixel - 0 - 1);
	s_arr[1][1] = (pixel - 0 - 0);
	s_arr[1][2] = (pixel - 0 + 1);
	s_arr[2][0] = (pixel + num_of_cols - 1);
	s_arr[2][1] = (pixel + num_of_cols - 0);
	s_arr[2][2] = (pixel + num_of_cols + 1);

	float V = 0;
	for (unsigned int i = 0; i < 3; i++) {
		for (unsigned int j = 0; j < 3; j++) {
			if (i==1 && j==1) continue;
			float Vxz = 0.0;
			unsigned int px = s_arr[1][1]->period;
			unsigned int pz = s_arr[i][j]->period;
			unsigned int T = lcm(p, pz);
			for (unsigned int t = 0; t < T; t++) {
				unsigned int sx = s_arr[1][1]->stage1_s_arr[p];
				unsigned int sz = s_arr[i][j]->stage1_s_arr[pz];
				int Vx = (s_arr[1][1]->pixel_values[phi(sx, p, t)] - s_arr[1][1]->pixel_values[phi(sz, pz, t)]);
				int Vz = (s_arr[i][j]->pixel_values[phi(sx, p, t)] - s_arr[i][j]->pixel_values[phi(sz, pz, t)]);
				/*if (px != 1 && pz != 1) {
					cout << T;
					cout << "PP:" << px << "#" << pz << "SS:" << sx << "#" << sz << "TT" << t << "++" << phi(sx, px, t) << "##" << phi(sz, pz, t) << endl;
					getchar();
				}*/
				Vxz += (Vx*Vx) + (Vz*Vz) + (float)10.0*(abs((float)p - (float)pz));
			}
			Vxz /= T;
			float gammaxz = (1 / (1 + (100 * (float)s_arr[1][1]->mad_vals[i][j])));
			V += (Vxz * gammaxz);
		}
	}
	return V;
}

int main(int ac, char** av)
{

	//Code before integration
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
          /*for(int k=0;k<100;k++)
          {
        	  printf("\n %d",p_sorted.at(k));
          }*/

      // Segmentation ends
          // Add loop time and start time finding code here


/*
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
*/
          printf("\nIntegrated code starting..\n");
  //Integrated code starts
          int ip_frame_num = 0;
          	Mat input_frame;
          	int num_of_rows, num_of_cols;
          	ofstream myfile;

          	myfile.open("period.txt", ios::out);
          	VideoCapture capture("SamplePoolPalms.mp4");
          	if (!capture.isOpened()) {
          		printf("Failed to open a video device\n");
          		return 1;
          	}
          	for (ip_frame_num = 0; ip_frame_num < NO_OF_FRAMES; ip_frame_num++) {
          		capture >> input_frame;
          		if (input_frame.empty())
          			break;
          		input_frame.copyTo(input_video_bgr[ip_frame_num]);
          		cvtColor(input_frame, input_video[ip_frame_num], CV_BGR2GRAY);
          		num_of_rows = input_frame.rows;
          		num_of_cols = input_frame.cols;
          	}
          	unsigned int num_of_frames = ip_frame_num;

          	unsigned int start_row = 0;
          	//num_of_rows = 1088;

          	unsigned int start_col = 0;
          	//num_of_cols = 1920;

          	xPixel * pixels = new xPixel[(num_of_rows + 1)*(num_of_cols + 1)];
          	{
          		xPixel * pixels_itr = pixels;
          		for (unsigned int row = start_row; row < start_row + num_of_rows; row++) {
          			for (unsigned int col = start_col; col < start_col + num_of_cols; col++) {
          				for (unsigned int ip_frame_num = 0; ip_frame_num < num_of_frames; ip_frame_num++) {
          					pixels_itr->pixel_values[ip_frame_num] = input_video[ip_frame_num].at<unsigned char>(row, col);
          				}
          				pixels_itr++;
          			}
          		}
          		pixels_itr = pixels + num_of_cols + 1;
          		for (unsigned int row = start_row; row < start_row + num_of_rows - 1; row++) {
          			for (unsigned int col = start_col; col < start_col + num_of_cols - 1; col++) {
          				pixels_itr->calc_variance(num_of_frames, num_of_cols);
          				pixels_itr++;
          			}
          		}

          	}

          	printf("Starting Stage 1\n");

          	//STAGE1
          	for (unsigned int p = 40; p < (num_of_frames / 2); p+=4) {
          		printf("Seeding s\n");
          		//get initial estimate for s for p i.e. L|p
          		xPixel * pixels_itr = pixels;
          		for (unsigned int row = start_row; row < start_row + num_of_rows; row++) {
          			for (unsigned int col = start_col; col < start_col + num_of_cols; col++) {
          				float min_etemporal = 1000000;
          				unsigned int min_s = 0;
          				for (unsigned int s = 1; s < ((num_of_frames/2)-1); s += 4) {
          					float pixel_etemporal = get_pixel_etemporal(pixels_itr, num_of_frames, s, p);
          					if (pixel_etemporal < min_etemporal) {
          						min_etemporal = pixel_etemporal;
          						min_s = s;
          					}
          				}
          				pixels_itr->stage1_s_arr[p] = min_s;
          				pixels_itr++;
          			}
          		}
          	}
          	printf("Starting Stage 2\n");

          	//STAGE2
          	printf("Seeding p\n");
          	//select inital p
          	{
          		xPixel * pixels_itr = pixels + num_of_cols + 1;
          		for (unsigned int row = start_row; row < start_row + num_of_rows - 1; row++) {
          			for (unsigned int col = start_col; col < start_col + num_of_cols - 1; col++) {
          				unsigned int min_p = 0;
          				float min_E = 100000;
          				for (unsigned int p = 40; p < (num_of_frames / 2); p += 4) {
          					float ep = pixels_itr->stage1_e_arr[p];
          					if (ep < min_E) {
          						min_E = ep;
          						min_p = p;
          					}
          				}

          				if (pixels_itr->is_static) {
          					pixels_itr->period = 1;
          				} else {
          					pixels_itr->period = min_p;
          				}
          				pixels_itr++;
          			}
          		}
          	}
          	printf("Starting period optimization\n");
          	//optimize p using simulated alpha-expansion
          	for (int z = 0; z < 1; z++) {
          		printf("z = %d\n", z);
          		xPixel * pixels_itr = pixels + num_of_cols + 1;
          		for (unsigned int row = start_row; row < start_row + num_of_rows - 1; row++) {
          			printf("row = %d\n", row);
          			for (unsigned int col = start_col; col < start_col + num_of_cols - 1; col++) {
          				if (pixels_itr->is_static == 1) {
          					pixels_itr->start = 0;
          					pixels_itr->new_period = 1;
          					pixels_itr++;
          					continue;
          				}
          				unsigned int min_s = 0;
          				unsigned int min_p = 10;
          				float min_E = 100000;
          				unsigned int init_val = pixels_itr->period;
          				for (unsigned int p = 40; p < (num_of_frames / 2); p += 4) {
          					float pixel_espatial = get_pixel_espatial_stage2(pixels_itr,num_of_frames, num_of_cols, p);
          					float pixel_etemporal = get_pixel_etemporal(pixels_itr, num_of_frames, pixels_itr->stage1_s_arr[p], p);
          					float E = 10 * pixel_espatial + pixel_etemporal;
          					if (E < min_E) {
          						min_E = E;
          						min_p = p;
          						min_s = pixels_itr->stage1_s_arr[p];
          					}
          				}
          				pixels_itr->start = min_s;
          				pixels_itr->new_period = min_p;
          				//myfile << init_val << "->" << min_p << "\t\t";
          				pixels_itr++;
          			}
          			//myfile << endl;
          		}
          		pixels_itr = pixels + num_of_cols + 1;
          		for (unsigned int row = start_row; row < start_row + num_of_rows - 1; row++) {
          			for (unsigned int col = start_col; col < start_col + num_of_cols - 1; col++) {
          				pixels_itr->period = pixels_itr->new_period;
          				pixels_itr++;
          			}
          		}
          		pixels_itr = pixels;
          		for (unsigned int row = start_row; row < start_row + num_of_rows - 1; row++) {
          			for (unsigned int col = start_col; col < start_col + num_of_cols - 1; col++) {
          				//myfile << int(pixels_itr->period) << "\t";
          				pixels_itr++;
          			}
          			//myfile << endl;
          		}
          		//myfile << "####################################################" << endl;
          		//myfile << "####################################################" << endl;
          		//myfile << "####################################################" << endl;
          	}

          	VideoWriter launch, launch_bgr;
          	//launch.open("C:\\Users\\Nikhil\\Desktop\\Stanford\\cs231\\project\\op.avi", CV_FOURCC('D', 'I', 'V', 'X'), 30, Size(num_of_cols, 2 * num_of_rows), false);
          	launch_bgr.open("out.avi", CV_FOURCC('D', 'I', 'V', 'X'), 30, Size(num_of_cols, num_of_rows), true);

          	if (!launch_bgr.isOpened())
          	{
          		printf("Could not open the output video for write:\n");
          		getchar();
          		return -1;
          	}

          	//Mat output_frame;
          	Mat output_frame_bgr = imread("1.bmp",CV_LOAD_IMAGE_COLOR);
          	//Mat output_frame_bgr(num_of_rows, num_of_cols, CV_8UC3);;
          	//unsigned char* frame_buffer = new unsigned char[(2 * num_of_rows)*num_of_cols];
          	int row,col;
          	for (unsigned int op_frame = 0; op_frame < NO_OF_FRAMES * 4; op_frame++) {
          		xPixel * pixels_itr = pixels;
          		//unsigned char* frame_itr = frame_buffer;
          				for (int k=0; k<p_sorted.at(0);k++)
          				{
          					pixels_itr++;

          				}
          				//printf("\nIter increased %d times to correct for starting value",p_sorted.at(0));

          				for (int k=0;k<p_sorted.size();k++)
          				{
          					//printf("\nEntered %d\n",k);
          					row=p_sorted.at(k)/num_of_cols;
          					col=p_sorted.at(k)%num_of_cols;
          					unsigned int p = pixels_itr->period;
          					unsigned int s = pixels_itr->stage1_s_arr[p];
          					unsigned int phi_val = phi(s, p, op_frame);
          					Vec3b intensity = input_video_bgr[phi_val].at<Vec3b>(row, col);
          					output_frame_bgr.at<Vec3b>(row, col) = intensity;
          					if(k!=(p_sorted.size()-1))
          					{
          						for(int j=0;j<(p_sorted.at(k+1)-p_sorted.at(k));j++)
          						{
          							pixels_itr++;
          						}
          					}
          					//printf("\nExited %d\n",k);
          				}

          				//*frame_itr = pixels_itr->pixel_values[phi_val];
          				//*(frame_itr+(num_of_rows*num_of_cols)) = input_video[(op_frame % num_of_frames)].at<unsigned char>(row, col);
          				/*for(int j=0;j<selected_pixel_indices.size();j++)
          				{
          					if(selected_pixel_indices.at(j)==(num_of_cols*row+col))
          					{

          					}
          				}*/
          				//frame_itr++;

          		//output_frame = Mat(2*num_of_rows, num_of_cols, CV_8UC1, frame_buffer);
          		//launch.write(output_frame);
          		launch_bgr.write(output_frame_bgr);
          	}
          	myfile.close();
          	printf("\nDone!\n");
          	getchar();
          	return 0;
}
