#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

const int fast_threshold = 20; 

// Function to check if a pixel is a corner using FAST
bool isCorner(const Mat& img, int x, int y) {
    int pixelValue = img.at<uchar>(y, x);

    const int offsets[16][2] = {
        {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
        {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}
    };

    int brighter = 0, darker = 0;

    for (int i = 0; i < 16; ++i) {
        int newX = x + offsets[i][0];
        int newY = y + offsets[i][1];

        if (newX >= 0 && newX < img.cols && newY >= 0 && newY < img.rows) {
            if (img.at<uchar>(newY, newX) - pixelValue >= fast_threshold) {
                brighter++;
            } else if (pixelValue - img.at<uchar>(newY, newX) >= fast_threshold) {
                darker++;
            }
        }
    }

    return (brighter >= 9 || darker >= 9); // Adjusted condition for corner detection
}

vector<Point2i> detectCornersFAST(const Mat& img) {
    vector<Point2i> corners;

    for (int y = 3; y < img.rows - 3; ++y) {
        for (int x = 3; x < img.cols - 3; ++x) {
            if (isCorner(img, x, y)) {
                corners.push_back(Point2i(x, y));
            }
        }
    }

    return corners;
}

Mat OpenCV_Fast(Mat inputImage)
{
    // OpenCV FAST corner detection
    Mat Image = inputImage;
    vector<KeyPoint> keypoints;
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    detector->detect(Image, keypoints);

    // Display detected corners
    Mat cornersImageCV;
    drawKeypoints(Image, keypoints, cornersImageCV, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("Detected Corners (FAST_CV)", cornersImageCV);
    waitKey(0);

    // Convert keypoints to Points
    vector<Point2f> points;
    for (size_t i = 0; i < keypoints.size(); ++i) {
        points.push_back(keypoints[i].pt);
    }
    return cornersImageCV;
}

void MyFast(Mat InputImage)
{
    //My FAST
    Mat Image = InputImage;
    vector<Point2i> detectedCorners = detectCornersFAST(Image);

    if (detectedCorners.empty()) {
        cout << "No corners detected!" << endl;
        return;
    }

    Mat cornersImage;
    cvtColor(Image, cornersImage, COLOR_GRAY2BGR);

    for (const Point2i& corner : detectedCorners) {
        circle(cornersImage, corner, 3, Scalar(0, 255, 0), 2);
    }

    imshow("Detected Corners (FAST)", cornersImage);
    waitKey(0);
}

void OpenCV_KLT(string path)
{
     VideoCapture video(path);

    // Check if the GIF is opened successfully
    if (!video.isOpened()) {
        cerr << "Could not open the GIF." << endl;
        return;
    }

    // Read the first frame
    Mat prevFrame, prevGray;
    video >> prevFrame;

    // Convert the first frame to grayscale
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

    // Detect FAST keypoints in the first frame
    vector<KeyPoint> keypoints;
    int fastThreshold = 20; // Adjust this threshold as needed
    bool nonMaxSuppression = true;
    FAST(prevGray, keypoints, fastThreshold, nonMaxSuppression);

    // Create a vector of points to store keypoint positions
    vector<Point2f> prevPoints;
    for (const auto& kp : keypoints) {
        prevPoints.emplace_back(kp.pt);
    }

    Mat frame, gray;
    vector<Point2f> nextPoints;
    vector<uchar> status;
    vector<float> err;

    while (true) {
        // Read the next frame
        video >> frame;

        // Check if the video ends
        if (frame.empty()) {
            break;
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Calculate optical flow using KLT
        calcOpticalFlowPyrLK(prevGray, gray, prevPoints, nextPoints, status, err);

        // Filter out keypoints for which the KLT tracking failed or lost track
        size_t i = 0;
        for (i = 0; i < prevPoints.size(); ++i) {
            if (status[i]) {
                line(frame, prevPoints[i], nextPoints[i], Scalar(0, 0, 255), 2);
                circle(frame, nextPoints[i], 5, Scalar(0, 255, 0), -1);
            }
        }

        // Update the previous frame and keypoints for the next iteration
        prevGray = gray.clone();
        prevPoints = nextPoints;

        // Show the frame with keypoints and tracked points
        imshow("KLT Tracking", frame);

        // Exit if the 'Esc' key is pressed
        char key = waitKey(30);
        if (key == 27) {
            break;
        }
    }

    video.release();
    destroyAllWindows();
}

int main(int, char**){

    Mat inputImageFAST_1 = imread("/FAST/signal-2023-12-14-212155_002.jpeg", IMREAD_GRAYSCALE);

    if (inputImageFAST_1.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }
    
    Mat inputImageFAST_2 = imread("/FAST/signal-2023-12-14-212155_003.jpeg", IMREAD_GRAYSCALE);

    if (inputImageFAST_2.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    OpenCV_Fast(inputImageFAST_1);
    MyFast(inputImageFAST_1);

    OpenCV_Fast(inputImageFAST_2);
    MyFast(inputImageFAST_2);

    string path = "/KLT/4.gif";
    OpenCV_KLT(path);
    

    return 0;
}