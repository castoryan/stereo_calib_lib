#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "spherical_multiple_filter_stereo_calib_lib.h"
#include "images/imagesBase.h"
#include <iostream>

#include "features/featuresSIFT.h"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char * argv[])
{
    //this are the intrinsic parameters of your cameras before rectification
    imagesBase_initial_parameters iip;
    iip.left_resx = 1280;
    iip.left_resy = 720;
    iip.left_cx = 625.38258;
    iip.left_cy = 405.82695;
    iip.left_fx = 1300.16067;
    iip.left_fy = 1300.81054;
    iip.left_k1  = -0.51084;
    iip.left_k2  = 0.43363;
    iip.left_k3  = 0.0;
    iip.left_p1  = 0.0;
    iip.left_p2  = 0.0;

    iip.right_resx = 1280;
    iip.right_resy = 720;
    iip.right_cx = 622.67895;
    iip.right_cy = 455.41158;
    iip.right_fx = 1300.82627;
    iip.right_fy = 1300.72700;
    iip.right_k1  = -0.52108;
    iip.right_k2  = 0.4922;
    iip.right_k3  = 0.0;
    iip.right_p1  = 0.0;
    iip.right_p2  = 0.0;

    imagesBase ib(iip);

    //capture the first images from your cameras
    //VideoCapture cap1, cap2;
    //cap1.open(0);
    //cap2.open(1);

    Mat leftraw, rightraw, left, right;
    char buf[100];
    sprintf(buf,"/home/castoryan-home/data_hobot/05/image_0/%06d.png",0);
    cout<<buf<<endl;
    left = cv::imread(buf,-1);
    cout<<left.size()<<endl;
    
    sprintf(buf,"/home/castoryan-home/data_hobot/05/image_1/%06d.png",0);
    cout<<buf<<endl;
    right = cv::imread(buf,-1);
    cout<<right.size()<<endl;
    //cap2 >> right;
    //cap1 >> left;

    //rectify your images. Get the new intrinsic parameters that should be sent to the stereo calibration system.
    
    imagesBase_data ibd = ib.rectify(left, right);
    Mat kleft = ibd.calibMatLeft;
    Mat kright = ibd.calibMatRight;
    int width = iip.left_resx;
    int height = iip.left_resy;
    

    cout<<"setting images"<<endl;
    cout<<width<<"  "<<height<<endl;
    double resize_factor = 1;

    //set the parameters for the stereo calibration system
    spherical_multiple_filter_stereo_calib_params cscp_general;
    cscp_general.baseline = 540;//in mm
    cscp_general.left_cam_resx = width/resize_factor;
    cscp_general.left_cam_resy = height/resize_factor;
    cscp_general.left_cam_cx = kleft.at<double>(0,2)/resize_factor;
    cscp_general.left_cam_cy = kleft.at<double>(1,2)/resize_factor;
    cscp_general.left_cam_fx = kleft.at<double>(0,0)/resize_factor;
    cscp_general.left_cam_fy = kleft.at<double>(1,1)/resize_factor;
    cscp_general.right_cam_resx = width/resize_factor;
    cscp_general.right_cam_resy = height/resize_factor;
    cscp_general.right_cam_cx = kright.at<double>(0,2)/resize_factor;
    cscp_general.right_cam_cy = kright.at<double>(1,2)/resize_factor;
    cscp_general.right_cam_fx = kright.at<double>(0,0)/resize_factor;
    cscp_general.right_cam_fy = kright.at<double>(1,1)/resize_factor;

    //cout<<kleft<<endl;
    //cout<<kright<<endl;
    spherical_multiple_filter_stereo_calib csc(cscp_general);
    

    cout<<"Starting main Loop!!!"<<endl;
    int count=0;
    while(1)
    {
        count++;
        char buf[100];
        sprintf(buf,"/home/castoryan-home/data_hobot/20171223_img/image_1/%06d.png",count);
        left = cv::imread(buf,-1);
        cout<<left.size()<<endl;
        
        sprintf(buf,"/home/castoryan-home/data_hobot/20171223_img/image_0/%06d.png",count);
        right = cv::imread(buf,-1);
        cout<<right.size()<<endl;
        

	imagesBase_data ibd = ib.rectify(left, right);

	Mat left_rz, right_rz;

	resize(ibd.rectifiedLeftImage, left_rz, Size(left.cols/resize_factor,left.rows/resize_factor));
        resize(ibd.rectifiedRightImage, right_rz, Size(right.cols/resize_factor,right.rows/resize_factor));

	csc.calibrate(left_rz, right_rz);
        //csc.calibrate(left, right);

	//get the calibrated transformations between the two cameras
	spherical_multiple_filter_stereo_calib_data cscd =  csc.get_calibrated_transformations();

	//obtain and show the disparity map
        cv::Mat csdd = csc.get_disparity_map(left_rz, right_rz);
        imshow("disparity", csdd);
	waitKey(10);

	//show the transformation between the left and right images
	cout << "Transformation from left to right camera: " << cscd.transformation_left_cam_to_right_cam << endl;
        

    }

    return 0;
}
