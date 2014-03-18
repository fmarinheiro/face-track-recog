#include <stdio.h>
#include <iostream>
#include <string>
#include "functions.h"


using namespace std;

int main(int argc, char** argv)
{

	IplImage *frame;
	IplImage *result_frame;
	IplImage *detected_face;
	IplImage *detected_face_processed;

	// Structure for getting video from camera or avi
	CvCapture* capture = 0;

	// Haar Cascade file, used for Face Detection.
	char *faceCascadeFilename = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";

	// Load the HaarCascade classifier for face detection.
	CvHaarClassifierCascade* faceCascade;
	faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0);

	if( !faceCascade ) {
		printf("Couldnt load Face detector '%s'\n", faceCascadeFilename);
		exit(1);
	}

	cvNamedWindow( "window 1" );
    cvNamedWindow( "window 2" );
    cvNamedWindow( "window 3" );

	// Capture from the camera.
	capture = cvCaptureFromCAM(0);

	for(;;)
	{
		// Capture the frame and load it in IplImage
		if( !cvGrabFrame( capture ))
			break;

		frame = cvRetrieveFrame( capture );

		// If the frame does not exist, quit the loop
		if( !frame )
			break;

		// Allocate framecopy as the same size of the frame
		if( !result_frame )
			result_frame = cvCreateImage( cvSize(frame->width,frame->height),
										IPL_DEPTH_8U, frame->nChannels );

		// Check the origin of image. If top left, copy the image frame to frame_copy.
		if( frame->origin == IPL_ORIGIN_TL )
			cvCopy( frame, result_frame, 0 );
		// Else flip and copy the image
		else
			cvFlip( frame, result_frame, 0 );

		// Perform face detection on the input image, using the given Haar classifier
	    CvRect faceRect = detectFaceInImage(frame, faceCascade);

	    cvShowImage("window 1", frame);

	    if(0 > faceRect.x || 0 > faceRect.y)
	    {
	    	cvShowImage("window 2", frame);
	    }
	    else
	    {

	    	cvRectangle(result_frame, cvPoint(faceRect.x, faceRect.y),  cvPoint(faceRect.x + faceRect.width, faceRect.y + faceRect.height), CV_RGB(255,255,255));
	        cvShowImage("window 2", result_frame);

	        cvSetImageROI(frame, faceRect);

	        detected_face = cvCreateImage(cvGetSize(frame), frame->depth, frame->nChannels);

            cvCopy(frame, detected_face);

            cvResetImageROI(frame);

            detected_face_processed = processImage(detected_face);

	        cvShowImage("window 3", detected_face_processed);

	    }

	    if(result_frame)
	    	cvReleaseImage(&result_frame);

	    if(detected_face_processed)
	    	cvReleaseImage(&detected_face_processed);

	    if (detected_face)
	    	cvReleaseImage(&detected_face);


		// Wait for a while before proceeding to the next frame
		if( cvWaitKey( 10 ) >= 0 )
			break;
	}

    // release the image
	cvReleaseImage(&frame);
	cvReleaseImage(&result_frame);
	cvReleaseImage(&detected_face);
	cvReleaseImage(&detected_face_processed);
	cvDestroyAllWindows();

	cvWaitKey(0);

	return 0;

}
