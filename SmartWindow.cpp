#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>

#pragma omp parallel for

#include <opencv2/highgui/highgui.hpp>
#include <wtypes.h>


using namespace std;
using namespace cv;

#define ImageX 1200
#define ImageY 600
#define WindowX 1080
#define WindowY 720


///////////////
// 문자열 버퍼 크기 정의
const int MAX_STR_BUFFER_SIZE = 32;

// 시작 시간
DWORD startTime = 0;

// 전체 Frame 수
long FrameCount = 0;
long LostCount = 0;
float fpssum = 0;
///////////////////////////////


//웹캠 해상도 640*480
//윈도우 크기 1080*720
/* @ function main */

#ifdef _DEBUG
#pragma comment(lib,"winmm.lib")//디버그 모드인 경우
#else
#pragma comment(lib,"winmm.lib")//릴리즈 모드인 경우
#endif


int main(int argc, char* argv[])
{
	///////////////////////fps code start
 // 카메라 프레임을 저장할 이미지 변수
	IplImage* image_ = NULL;

	// 카메라를 연다
	//CvCapture* capture = cvCaptureFromCAM(0);

	// Resize 가능한 윈도우 생성
	//cvNamedWindow("Camera", CV_WINDOW_NORMAL);

	// FPS를 표시해 줄 Font 및 문자열 버퍼 초기화
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, 1.0);

	char strBuffer[MAX_STR_BUFFER_SIZE] = { 0, };//문자열 버퍼 초기화

	// 시간정보 초기화
	startTime = timeGetTime();

	///////////////fpscodeend

	Mat frame_;
	cv::Mat grayframe;

	//fps표시기준점
	cv::Point myPoint;
	myPoint.x = 10;
	myPoint.y = 40;

	//얼굴 위치 인식 변수
	double face_x = 0;
	double face_y = 0;
	int distance = 0;

	//fps
	float fps = 0;



	Mat subimage;

	// open the default camera
	cv::VideoCapture cap(0);

	// check if we succeeded
	if (!cap.isOpened()) {
		std::cerr << "Could not open camera" << std::endl;
		return -1;
	}

	// create a window
	cv::namedWindow("webcam", 1);
	cv::namedWindow("window", 1);

	// face detection configuration
	cv::CascadeClassifier face_classifier;
	face_classifier.load("haarcascade_frontalface_default.xml");

	cv::Mat frame, resized_image;


	//Create videocapture object to read from video file
	
	VideoCapture cap_("sample720.mp4");

	//check if the file was opened properly
	if (!cap_.isOpened()) {
		cout << "Capture could not be opened succesfully" << endl;
		return -1;
	}

	while (1) {
		DWORD timetemp = timeGetTime();//////////////\\\\

		bool frame_valid = true;

		cap_ >> frame_;


		if (frame_.empty()) {
			cout << "Video over" << endl;
			break;
		}

		try {
			// get a new frame from webcam
			cap >> frame;
		}
		catch (cv::Exception & e) {
			std::cerr << "Exception occurred. Ignoring frame... " << e.err << std::endl;
			frame_valid = false;
		}

		if (frame_valid) {
			try {

				cv::cvtColor(frame, grayframe, CV_BGR2GRAY);//frame to gray scale
				cv::equalizeHist(grayframe, grayframe);//equalize


				// a vector array to store the face found//face당 1개 생성
				std::vector<cv::Rect> faces;
				
				if (distance > 200)
				{
					face_classifier.detectMultiScale(grayframe, faces,
						1.1, // increase search scale by 10% each pass
						3,   // merge groups of three detections
						CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
						cv::Size(120, 120)
					);

				}
				else
				{
					face_classifier.detectMultiScale(grayframe, faces,
						1.1, // increase search scale by 10% each pass
						3,   // merge groups of three detections
						CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
						cv::Size(30, 30)
					);

				}
				
				/*
				
				face_classifier.detectMultiScale(grayframe, faces,
				   1.1, // increase search scale by 10% each pass
				   3,   // merge groups of three detections
				   CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
				   cv::Size(30, 30)//////######초기 30, 30
				);
				*/
				

				// -------------------------------------------------------------
				// draw the results
				for (int i = 0; i < faces.size(); i++) {
					cv::Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
					cv::Point tr(faces[i].x, faces[i].y);

					face_x = faces[i].x + (faces[i].width / 2);
					face_y = faces[i].y + (faces[i].height / 2);//얼굴 중심점 xy좌표
					distance = sqrt((faces[i].width * faces[i].width) + (faces[i].height * faces[i].height)); //얼굴 사각형 대각선 길이

					Rect rect(faces[i].x + (ImageX / 10), faces[i].y + (ImageY / 10), distance, distance);//이미지 크기 자르는 부분

					//subimage = image(rect);
					subimage = frame_(rect);

					//cv::rectangle(frame, lb, tr, cv::Scalar(0, 255, 0), 3, 4, 0);
				}



				// print the output

				cv::resize(subimage, resized_image, Size(WindowX, WindowY), 0, 0, CV_INTER_LINEAR);//이미지 크기 조절            


				FrameCount++;

				// 누적 FPS를 계산한다.
				fps = (float)1000.0 / (timeGetTime() - timetemp);
				float avefps = (float)((FrameCount * 1000.0) / (timeGetTime() - startTime));

				// FPS를 이미지 버퍼에 출력한다.
				sprintf_s(strBuffer, "fps: %.2lf avefps : %.2lf", fps, avefps);
				cv::putText(resized_image, strBuffer, myPoint, 2, 2, Scalar::all(0));

				cv::imshow("window", resized_image);//이미지 출력
				//##cv::imshow("webcam", frame);////웹캠 테스트용 
				if (faces.size() == 0) {
					LostCount++;
				}

				fpssum += fps;
				cout << "TotalFrame : " << FrameCount << " LostFrame : " << LostCount << " Accuracy : " << (float)(FrameCount - LostCount) * 100 / FrameCount << "% avefps : " << fpssum / FrameCount << endl;
				faces.clear();//faces벡터초기화
			}
			catch (cv::Exception & e) {
				std::cerr << "Exception occurred. Ignoring frame... " << e.err << std::endl;
			}
		}

		if (cv::waitKey(1) >= 0) break;
	}

	// VideoCapture automatically deallocate camera object
	return 0;

}
