#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <fstream>
#include<math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/objdetect.hpp>
#define PI 3.1415926
/*蓝色
int hmin = 94, hmax = 118, smin = 144, smax = 255, vmin = 0, vmax = 255;
//黄色
int hmin1 = 17, hmax1 = 40, smin1 = 108, smax1 = 255, vmin1 = 50, vmax1 = 255;
//黑色
int hmin2 = 0, hmax2 = 180, smin2 = 0, smax2 = 255, vmin2 = 0, vmax2 = 40;
//红色
int hmin3 = 0, hmax3 = 7, smin3 = 172, smax3 = 255, vmin3 = 0, vmax3 = 255;*/
int hmin[4] = { 94,17,0,0 };
int hmax[4] = { 118,40,180,7 };
int smin[4] = { 144,108,0,172 };
int smax[4] = {255,255,255,255};
int vmin[4] = {0,50,0,0};
int vmax[4] = {255,255,40,255};
int a[4];

using namespace cv;
using namespace std;

class BallTracker {
private:
    KalmanFilter kf;
    Mat state;
    Mat prediction;
    Point prevCenter;
    Mat frame, canny;
    vector<vector<Point>> contours;
    int w;
    Point2f center;
    float radius;
    int maxArea;
public:
    BallTracker() :kf(4, 2, 0), w(0), radius(0), maxArea(0) {
        kf.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
        kf.measurementMatrix = (Mat_<float>(2, 4) << 1, 0, 0, 0,
            0, 1, 0, 0);
        kf.processNoiseCov = (Mat_<float>(4, 4) << 1e-5, 0, 0, 0,
            0, 1e-5, 0, 0,
            0, 0, 1e-5, 0,
            0, 0, 0, 1e-5) * 0.03;
        kf.measurementNoiseCov = (Mat_<float>(2, 2) << 1, 0,
            0, 1) * 0.1;
        state = (Mat_<float>(4, 1) << 0, 0, 0, 0);
    }
    int* findcontours(Mat canny, Mat img,Scalar lower,Scalar upper)
    {
        int maxContourIdx = -1;
        Mat canny1, binary;
        Mat hsv, mask;
        cvtColor(img, hsv, COLOR_BGR2HSV);
        inRange(hsv, lower, upper, mask);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(mask, binary, MORPH_CLOSE, kernel1);
        morphologyEx(binary, binary, MORPH_OPEN, kernel1);
        Canny(binary, canny1, 50, 150);
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        Point2f center1;
        float radius1 = 0;
        int b = 0;
        for (int i = 0;i < contours.size();i++)
        {
            minEnclosingCircle(contours[i], center1, radius1);
            int area = contourArea(contours[i]);
            if (area > 0.7 * PI * radius1 * radius1)
            {
                if (area > b)
                {
                    b = area;
                    maxContourIdx = i;
                    radius = radius1;
                    center = center1;
                }
            }
        }
        maxArea = b;
        w = maxContourIdx;
        if (!contours.empty())
        {
            //circle(img, center, (int)radius, Scalar(0, 255, 0), 1, 0);
            Rect rect(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
            rectangle(img, rect, Scalar(255, 0, 255), 2);
            //circle(img, center, 10, Scalar(255, 0, 255), 1, 0);
            a[0] = center.x - radius;
            a[1] = center.y - radius;
            a[2] = center.x + radius;
            a[3] = center.y + radius;
        }
        return a;
    }
    Point2f predicted(Mat frame) {
        prediction = kf.predict();
        //circle(frame, Point(prediction.at<float>(0), prediction.at<float>(1)), radius, Scalar(0, 255, 0), 1);
        Rect rect(prediction.at<float>(0) - radius, prediction.at<float>(1) - radius, 2 * radius, 2 * radius);
        rectangle(frame, rect, Scalar(255, 0, 255), 2);
        return Point(prediction.at<float>(0), prediction.at<float>(1));
    }
    Point2f measured(Mat frame) {

        Moments mu = moments(contours[w]);
        // 计算轮廓的中心
        Point center(mu.m10 / mu.m00, mu.m01 / mu.m00);
        // 预测小球位置
        Mat predicted = kf.predict();

        // 更新卡尔曼滤波器
        Mat measurement = (Mat_<float>(2, 1) << center.x, center.y);
        Mat corrected = kf.correct(measurement);

        // 在图像上绘制预测和校正后的位置
        //circle(frame, Point(predicted.at<float>(0), predicted.at<float>(1)), 5, Scalar(0, 0, 255), -1);
        //circle(frame, Point(corrected.at<float>(0), corrected.at<float>(1)), 5, Scalar(0, 255, 0), -1);
        // 更新上一帧的小球中心位置
        prevCenter = center;
        return Point(corrected.at<float>(0), corrected.at<float>(1));
    }
};
Mat distant(Mat canny, Mat image, int* a)
{
    vector<Point2d> image_points;
    image_points.push_back(Point2d(a[0], a[1]));
    image_points.push_back(Point2d(a[2], a[1]));
    image_points.push_back(Point2d(a[2], a[3]));
    image_points.push_back(Point2d(a[0], a[3]));

    //单位是mm
    std::vector<Point3d> model_points;
    model_points.push_back(Point3d(-20.0f, -20.0f, 0));
    model_points.push_back(Point3d(+20.0f, -20.0f, 0));
    model_points.push_back(Point3d(+20.0f, +20.0f, 0));
    model_points.push_back(Point3d(-20.0f, +20.0f, 0));

    Mat camera_matrix = (Mat_<double>(3, 3) << 972.2026772173228, 0, 133.2002102089862,
        0, 934.849349584515, 383.7714684971774,
        0, 0, 1);
    Mat dist_coeffs = (Mat_<double>(5, 1) << 0.1134925286013088, -0.01854931727807318, 0.01169435384640474, -0.04829855752982315, 0.2595514546035786);    
    Mat rotation_vector; 
    Mat translation_vector;
    solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector, 0, cv::SOLVEPNP_ITERATIVE);
    Mat Rvec;
    Mat_<float> Tvec;
    rotation_vector.convertTo(Rvec, CV_32F);  
    translation_vector.convertTo(Tvec, CV_32F); 
    Mat_<float> rotMat(3, 3);
    Rodrigues(Rvec, rotMat);
    Mat P_oc;
    P_oc = -rotMat.inv() * Tvec;
    float value = abs(P_oc.at<float>(2, 0));//0,0是离中轴线的垂直距离
    string text = to_string(value);
    putText(image, text, image_points[0], FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 255, 0), 2);
    return P_oc;

}
int main()
{
    VideoCapture cap(0);
    Mat frame, canny, gray, blur, hsv, dil, ero, binary;
    BallTracker kalman;
    while (1)
    {
        cap >> frame;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(frame, binary, MORPH_CLOSE, kernel);
        morphologyEx(binary, binary, MORPH_OPEN, kernel);
        Canny(binary, canny, 50, 150);
        /*char key = waitKey(1);
        if(key=='l' || key=='L')
            distant(canny, frame, kalman.findcontours(canny, frame, Scalar(hmin[0], smin[0], vmin[0]), Scalar(hmax[0], smax[0], vmax[0])));
        else if(key=='h' || key=='H')
            distant(canny, frame, kalman.findcontours(canny, frame, Scalar(hmin[1], smin[1], vmin[1]), Scalar(hmax[1], smax[1], vmax[1])));
        else if(key=='b' || key=='B')
            distant(canny, frame, kalman.findcontours(canny, frame, Scalar(hmin[2], smin[2], vmin[2]), Scalar(hmax[2], smax[2], vmax[2])));
        else if(key=='r'|| key=='R')
            distant(canny, frame, kalman.findcontours(canny, frame, Scalar(hmin[3], smin[3], vmin[3]), Scalar(hmax[3], smax[3], vmax[3])));
        */
        cout<<distant(canny, frame, kalman.findcontours(canny, frame, Scalar(hmin[3], smin[3], vmin[3]), Scalar(hmax[3], smax[3], vmax[3])))<<endl;
        imshow("frame", frame);
        //imshow("canny", canny);
        waitKey(0);
    }
    cap.release();
    destroyAllWindows();
    return 0;
}