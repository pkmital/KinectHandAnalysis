#include "PointDrawer.h"
#include "XnVDepthMessage.h"
#include <XnVHandPointContext.h>
#include <vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "alglibmisc.h"
#include "alglibinternal.h"
#include "linalg.h"
#include "statistics.h"
#include "dataanalysis.h"
#include "specialfunctions.h"
#include "solvers.h"
#include "optimization.h"
#include "diffequations.h"
#include "fasttransforms.h"
#include "integration.h"
#include "interpolation.h"

using namespace alglib;

// return 1/0 if val is within range of min/max
#define clip(val, min, max) ((val) < (max) && (val) > (min))

#define USE_GLUT 1

#ifdef USE_GLUT
	#if (XN_PLATFORM == XN_PLATFORM_MACOSX)
		#include <GLUT/glut.h>
	#else
		#include <GL/glut.h>
	#endif
#else
	#include "opengles.h"
#endif

#include "ofxOsc.h"

#define HOST "10.0.2.7"
#define PORT 12345

bool bSendOSC = true;

ofxOscSender sender;


void XnVPointDrawer::resampleVector(vector<Point> vec, vector<Point> &newVec, size_t newSize)
{
	size_t oldSize = vec.size();
	float ratio = oldSize/ (float)newSize;
	for(size_t i = 1; i < newSize-1; i++)
	{
		Point pt;
		pt.x = vec[floor((i-1) * ratio)].x * 0.25 + vec[floor((i) * ratio)].x * 0.5 + vec[floor((i+1) * ratio)].x * 0.25;
		pt.y = vec[floor((i-1) * ratio)].y * 0.25 + vec[floor((i) * ratio)].y * 0.5 + vec[floor((i+1) * ratio)].y * 0.25;
		newVec.push_back(pt);
	}
}


// Constructor. Receives the number of previous positions to store per hand,
// and a source for depth map
XnVPointDrawer::XnVPointDrawer(XnUInt32 nHistory, xn::DepthGenerator depthGenerator) :
	XnVPointControl("XnVPointDrawer"),
	m_nHistorySize(nHistory), m_DepthGenerator(depthGenerator), m_bDrawDM(false), m_bFrameID(false)
{
	m_pfPositionBuffer = new XnFloat[nHistory*3];
	
	hand_width = 130;
	hand_depth = 50; 
	hand_height = 130;
	hand_matrix = cv::Mat(hand_width, hand_height, CV_32F);
	
	prev_hand_matrix = cv::Mat(hand_size, hand_size, CV_32F);
	prev_hand_matrix.setTo(Scalar(0));
	hand_matrix_img = cv::Mat(hand_width, hand_height, CV_8UC3);
	hand_window_name = "Detected Hand";
	namedWindow(hand_window_name, CV_WINDOW_NORMAL);
	
	prev_handpt.X = 0;
	prev_handpt.Y = 0;
	prev_handpt.Z = 0;
	
	
	/*
	myMoments = (CvMoments*)malloc( sizeof(CvMoments) );  
    reset();   
    stor02 = cvCreateMemStorage(0);   
    stor03 = cvCreateMemStorage(0);   
	 */
	
	sender.setup(HOST, PORT);
}

// Destructor. Clear all data structures
XnVPointDrawer::~XnVPointDrawer()
{
	std::map<XnUInt32, std::list<XnPoint3D> >::iterator iter;
	for (iter = m_History.begin(); iter != m_History.end(); ++iter)
	{
		iter->second.clear();
	}
	m_History.clear();

	delete []m_pfPositionBuffer;
}

// Change whether or not to draw the depth map
void XnVPointDrawer::SetDepthMap(XnBool bDrawDM)
{
	m_bDrawDM = bDrawDM;
}
// Change whether or not to print the frame ID
void XnVPointDrawer::SetFrameID(XnBool bFrameID)
{
	m_bFrameID = bFrameID;
}

void XnVPointDrawer::SegmentHand(const XnPoint3D &ptHand)
{
	const XnDepthPixel (*DepthMatrix)[640] = NULL;
	unsigned int iTotalX = 0, iTotalY = 0;
	unsigned int iHalfX = 0, iHalfY = 0;
	int iBaseX = 0, iBaseY = 0;
	XnPoint3D ptHandProjection;
	xn::DepthMetaData DepthMD;
	int iX = 0, iY = 0;
	
	m_DepthGenerator.GetMetaData( DepthMD );
	
	DepthMatrix = (XnDepthPixel (*)[640])((XnDepthPixel*)DepthMD.Data());
		
	
	m_ptHandStart.X = ptHand.X - hand_width;
	m_ptHandEnd.X = ptHand.X + hand_width;
	m_ptHandStart.Y = ptHand.Y + hand_height;
	m_ptHandEnd.Y = ptHand.Y - hand_height;
	m_ptHandStart.Z = ptHand.Z;
	m_ptHandEnd.Z = ptHand.Z;
	
	m_DepthGenerator.ConvertRealWorldToProjective( 1, &ptHand, &ptHandProjection);
	m_DepthGenerator.ConvertRealWorldToProjective( 1, &m_ptHandStart, &m_ptHandStart);
	m_DepthGenerator.ConvertRealWorldToProjective( 1, &m_ptHandEnd, &m_ptHandEnd);
	
	
	//printf("%f - %f, %f - %f\n", m_ptHandStart.X, m_ptHandEnd.X, m_ptHandStart.Y, m_ptHandEnd.Y);
	
	iTotalX = abs(m_ptHandEnd.X - m_ptHandStart.X);
	iTotalY = abs(m_ptHandEnd.Y - m_ptHandStart.Y);
	
	iHalfX = iTotalX / 2;
	iHalfY = iTotalY / 2;
	
	printf("%d, %d\n", iTotalY, iTotalX);
	
	hand_matrix = Mat::zeros(iTotalY + 2, iTotalX + 2, CV_32F);
	
	iBaseX  = ((ptHandProjection.X - iHalfX) < 0) ? 0 : (ptHandProjection.X - iHalfX);
	iBaseY  = ((ptHandProjection.Y - iHalfY) < 0) ? 0 : (ptHandProjection.Y - iHalfY);
	
	for ( iY = iBaseY; (iY < ptHandProjection.Y + iHalfY) && (iY < 480); ++iY) 
	{
		for ( iX = iBaseX; (iX < ptHandProjection.X + iHalfX) && (iX < 640); ++iX) 
		{
			if ((DepthMatrix[iY][iX] > ptHandProjection.Z - hand_depth) &&
				(DepthMatrix[iY][iX] < ptHandProjection.Z + hand_depth))
			{
				hand_matrix.at<float>(iY - iBaseY, iX - iBaseX) = DepthMatrix[iY][iX] / 2048.0;
			}
		}
	}
	
	//dilate(hand_matrix, hand_matrix, Mat::ones(3,3,CV_32F));
	cv::resize(hand_matrix, scaled_hand_matrix, cv::Size(hand_size, hand_size));
	
	hand_matrix = scaled_hand_matrix * 0.8 + prev_hand_matrix * 0.2;
	prev_hand_matrix = hand_matrix.clone();
	
	dilate(hand_matrix, hand_matrix, Mat());
	erode(hand_matrix, hand_matrix, Mat());
	//dilate(hand_matrix, hand_matrix, Mat());
	//erode(hand_matrix, hand_matrix, Mat());
	blur(hand_matrix, hand_matrix, Size(iTotalX/20,iTotalX/20));
	erode(hand_matrix, hand_matrix, Mat());
	erode(hand_matrix, hand_matrix, Mat());
	erode(hand_matrix, hand_matrix, Mat());
	threshold(hand_matrix, hand_matrix, 0.05, 255, CV_THRESH_BINARY);
	
	
	hand_matrix.convertTo(hand_matrix, CV_8U);
	
	vector<vector<Point> > contours, hands;
    vector<Vec4i> hierarchy;
	Mat hand_matrix_clone = hand_matrix.clone();
    findContours(hand_matrix_clone, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	//Canny(hand_matrix, contourMat, 0.1, 0.8);
	cvtColor(hand_matrix, hand_matrix_img, CV_GRAY2RGB);
	
	
	cv::Moments mos = cv::moments(hand_matrix_clone, true);
	double hus[7];
	HuMoments(mos, hus);
	ofxOscMessage hu_msg;
	hu_msg.setAddress("/hu");
	for(int i = 0 ; i < 7; i++ )
		hu_msg.addFloatArg(hus[i]);
	sender.sendMessage(hu_msg);
	
	
	vector<Point> approx;
	
	printf("1");
	// test each contour
	for( size_t j = 0; j < contours.size(); j++ )
	{
		// approximate contour with accuracy proportional
		// to the contour perimeter
		
		Mat	contourMat = Mat(contours[j]);
		
		static int counter = 0;
		counter++;
		
		
		// low pass contour list:
		FILE *fp;
		char buf[256];
		sprintf(buf, "frame_%d.txt", counter);
		fp = fopen(buf, "w+");
		
		vector<Point> contoursCopy = contours[j];
		for( size_t i =1; i < contours[j].size()-1; i++)
		{
			contours[j][i].x = contoursCopy[i-1].x*0.25 + contoursCopy[i].x*0.5 + contoursCopy[i+1].x*0.25;
			contours[j][i].y = contoursCopy[i-1].y*0.25 + contoursCopy[i].y*0.5 + contoursCopy[i+1].y*0.25;
			
			fprintf(fp, "%d, %d\n", contours[j][i].x ,contours[j][i].y );
		}
		fclose(fp);
		

		
		
		
		/*
		void alglib::polynomialbuild(
									 real_1d_array x,
									 real_1d_array y,
									 barycentricinterpolant& p);
		
		
		barycentricinterpolant px, py;
		real_1d_array t,x,y;
		t.setlength(contours[j].size());
		x.setlength(contours[j].size());
		y.setlength(contours[j].size());
		double  *t_ptr = t.getcontent();
		double	*x_ptr = x.getcontent();
		double	*y_ptr = y.getcontent();
		for (size_t i = 0; i<contours[j].size(); i++) {
			t_ptr[i] = (double)i;
			x_ptr[i] = (double)contours[j][i].x;
			y_ptr[i] = (double)contours[j][i].y;
		}
		polynomialbuild(t, x, px);
		polynomialbuild(t, y, py);
		*/
		
		approxPolyDP(contourMat, approx, arcLength(Mat(contours[j]), true)*0.019, true);
		
		printf("2");
		//approxPolyDP(contourMat, approx, 20, true);
		
		//printf("arcLength(Mat(contours[j]), true)*0.02: %f", arcLength(Mat(contours[j]), true)*0.02);
		
		// distance threshold
		//approx.push_back(approx[0]);
		for (size_t i = 0; i < approx.size()-1; i++) {
			
			int idx = i; 
			int pdx = idx == 0 ? approx.size() - 1 : idx - 1; 
			int sdx = idx == approx.size() - 1 ? 0 : idx + 1; 
			
			Point v1 = approx[sdx] - approx[idx];
			Point v2 = approx[pdx] - approx[idx];
			
			float angle = acos( (v1.x*v2.x + v1.y*v2.y) / (norm(v1) * norm(v2)) );
			float distance = sqrt((approx[i].x-approx[i+1].x)*(approx[i].x-approx[i+1].x) + 
								  (approx[i].y-approx[i+1].y)*(approx[i].y-approx[i+1].y));
			if (distance < 30)//arcLength(Mat(contours[j]), true)*0.02*8) 
			{
				approx.erase(approx.begin()+i+1);
				i=0;
			}
			/*
			else if(angle > 2.0)
			{
				approx.erase(approx.begin()+i);
				i=0;
			}
			*/
		}
		
		// find contours with large area (to filter out noisy contours)
		// Note: absolute value of an area is used because
		// area may be positive or negative - in accordance with the
		// contour orientation
		if( approx.size() >= 4 && fabs(contourArea(Mat(approx))) > 3000 )
		{			 
			Scalar center = mean(contourMat);
			Point centerPoint = Point(center.val[0], center.val[1]);
			
			printf("3");
			
			//circle(hand_matrix_img, centerPoint, 5, Scalar(10, 10, 150), CV_FILLED, CV_AA);
			
			drawContours(hand_matrix_img, contours, 0, Scalar(10, 10, 200), 2, CV_AA);
			
			hands.push_back(approx);
			
			vector<int> hull;
			Mat handMat(hands[0]);
			cv::convexHull(handMat, hull, CV_CLOCKWISE);
			
			// draw approximated polygon as circles
			for( size_t i = 0; i < hands[0].size(); i++ )
			{
				circle(hand_matrix_img, hands[0][i], 2, Scalar(30, 120, 120), 1, CV_AA);
			}
			
			// draw approximated polygon as lines 
			for( size_t i = 0; i < hands[0].size()-1; i++ )
			{
				line(hand_matrix_img, hands[0][i], hands[0][i+1], Scalar(50, 120, 30), 1, CV_AA);
			}
			
			// draw the convex hull points
			vector<Point> hullPoints;
			for( size_t i = 0; i < hull.size(); i++ )
			{
				circle(hand_matrix_img, hands[0][hull[i]], 8, Scalar(10, 40, 100), 2, CV_AA);
				hullPoints.push_back(hands[0][hull[i]]);
			}
			
			// draw center of the the hand using convex hull (more accurate than the hand generator's position)
			Scalar centerHull = mean(Mat(hullPoints));
			Point centerHullPoint = Point(centerHull.val[0], centerHull.val[1]);
			
			circle(hand_matrix_img, centerHullPoint, 5, Scalar(20, 20, 100), 1, CV_AA);
			
			vector<Point> tips;
			vector<Point> nonTips;
			// fingertip detection (refinement of convex hull)
			// find interior angles of hull corners
			for( size_t i = 0; i < hull.size(); i++) 
			{
				int idx = hull[i]; 
				int pdx = idx == 0 ? approx.size() - 1 : idx - 1; 
				int sdx = idx == approx.size() - 1 ? 0 : idx + 1; 
				
				Point v1 = approx[sdx] - approx[idx];
				Point v2 = approx[pdx] - approx[idx];
				
				float angle = acos( (v1.x*v2.x + v1.y*v2.y) / (norm(v1) * norm(v2)) );
				
				// sharp angle must be finger
				if (angle < 1.0) 
				{	
					tips.push_back(approx[idx]);
					cv::circle(hand_matrix_img, approx[idx], 6, Scalar(200, 200, 200), 4, CV_AA);
				}
				else {
					nonTips.push_back(approx[idx]);
				}

			}
			
			
			printf("4");
			
			// calculate center of palm:
			if(handMat.rows > 0 )
			{
				//Scalar centerPalm = mean(Mat(nonTips));
				//Point centerPalmPoint = Point(centerPalm.val[0], centerPalm.val[1]);
				Point2f centerCircle;
				float radius;
				minEnclosingCircle(handMat, centerCircle, radius);
				circle(hand_matrix_img, centerCircle, radius, Scalar(20, 20, 20), 1, CV_AA);
				
			}
			
			
			
			vector<vector<Point> > defects;
			vector<vector<float> > defectDistances;
			vector<float> distances;
			// locate defects
			
			//printf("hull size: %d\n", hull.size());
			hull.push_back(hull[0]);
			for ( size_t i = 0; i < hull.size()-1; i++) 
			{
				//printf("hull[i]: %d, hull[i+1]: %d", hull[i], hull[i+1]);
				// look for points between the detected hull points:
				if (hull[i]+1 < hull[i+1]) 
				{
					vector<Point > defect;
					vector<float > distances;
					
					// median of hull end points
					Point v1 = (approx[hull[i]] + approx[hull[i+1]]);
					v1.x = v1.x/ 2.0;
					v1.y = v1.y/2.0;
					defect.push_back(v1);	 
					
					printf("5");
					size_t best_j = hull[i]+1;
					size_t best_distance = 0;
					// loop through in-between points (should just be one)
					for (size_t j = hull[i]+1; j <= hull[i+1]-1; j++) 
					{
						Point v2 = approx[j];
						line(hand_matrix_img, v1, v2, Scalar(200, 200, 200), 2, CV_AA);
						float distance = sqrt((v2.x-v1.x)*(v2.x-v1.x) + 
											  (v2.y-v1.y)*(v2.y-v1.y));
						if(distance > best_distance)
							best_j = j;
						
						//distances.push_back(distance);
						//printf("distance: %f\n", distance);
					}
					
					// store the maximum distanced point
					defect.push_back(approx[best_j]);
					
					// store the vector of median - max dist point
					defects.push_back(defect);
					
					// store the distances as floats
					defectDistances.push_back(distances);
				}
			}
			hull.pop_back();
			
			printf("6");
			/*
			vector<float> maximumDefectDistances; 
			for (size_t i = 0 ; i < defectDistances.size(); i++) 
			{
				//defects[i].clear();
				Point v1 = (approx[hull[i]] + approx[hull[i+1]]);
				v1.x = v1.x/ 2.0;
				v1.y = v1.y/2.0;
				//defect.push_back(v1);	
				
				int idx = i;
				
				maximumDefectDistances.push_back(defectDistances[i][0]);
				for (size_t k = 1; k < defectDistances[i].size(); k++)
				{
					if (defectDistances[i][k] > maximumDefectDistances[i])
					{
						maximumDefectDistances[i] = defectDistances[i][k];
					}
				}
			}
			*/
			
			// draw defects
			for (size_t i = 0; i < defects.size(); i++) {
				line(hand_matrix_img, defects[i][0], defects[i][1], Scalar(200,20,20), 2, CV_AA);
			}									 
			
			ofxOscMessage m,m2,m3,m4,m5,m6;
			m.setAddress("/hand/centroid");
			m.addIntArg(centerHull.val[0]);
			m.addIntArg(centerHull.val[1]);
			sender.sendMessage(m);
			
			
			const int MAX_HULL_POINTS = 14;
			const int MAX_DEFECT_POINTS = 8;
			
			approx.push_back(Point(0,0));
			while (hull.size() < MAX_HULL_POINTS) {
				hull.push_back(approx.size()-1);
			}
			/*
			while (maximumDefectDistances.size() < MAX_DEFECT_POINTS) {
				maximumDefectDistances.push_back(0);
			}
			*/
			int width=hand_matrix_img.cols;
			int height=hand_matrix_img.rows;
			
			
			/*
			m2.setAddress("/hand/point");
			for (size_t i = 0; i < hull.size(); i++) {
				m2.addFloatArg(approx[hull[i]].x / (float)width);
				m2.addFloatArg(approx[hull[i]].y / (float)height);
			}
			sender.sendMessage(m2);
			*/
			
			vector<float> xes, yes;
			int contour_size = contours[j].size();
			for (size_t i = 0; i < contour_size; i++) {
				xes.push_back(contours[j][i].x);
				yes.push_back(contours[j][i].y);
			}
			
			size_t npoints = 30;
			Mat contourMatX = Mat(xes);
			Mat contourMatY = Mat(yes);
			Mat contourMatXResampled,contourMatYResampled; 
			cv::resize(contourMatX, contourMatXResampled, Size(1,npoints), 0, 0, INTER_LINEAR);
			cv::resize(contourMatY, contourMatYResampled, Size(1,npoints), 0, 0, INTER_LINEAR);
			
			m2.setAddress("/hand/point");
			for (size_t i = 0; i < contourMatXResampled.rows; i++) {
				m2.addFloatArg(contourMatXResampled.at<float>(i) / (float)width);
				m2.addFloatArg(contourMatYResampled.at<float>(i) / (float)height);
			}
			sender.sendMessage(m2);
			
			
			m2.clear();
			m2.setAddress("/hand/numpoints");
			m2.addIntArg(hull.size());
			sender.sendMessage(m2);
			
			m3.setAddress("/hand/defect");
			/*
			for (size_t i = 0; i < defectDistances.size(); i++) {
				m3.addFloatArg(defectDistances[i]);
			}
			float max_dist = sqrt(width*width + height*height);
			for (size_t i = 0; i < maximumDefectDistances.size(); i++) {
				m3.addFloatArg( maximumDefectDistances[i] / max_dist);
			}
			sender.sendMessage(m3);
			 
			 */
			
			m4.setAddress("/hand/size");
			m4.addIntArg(hand_matrix_img.cols);
			m4.addIntArg(hand_matrix_img.rows);
			sender.sendMessage(m4);
										
			m5.setAddress("/hand/tips");
			for (size_t i = 0; i < tips.size(); i++) {
				m5.addFloatArg(tips[i].x);
				m5.addFloatArg(tips[i].y);
			}
			sender.sendMessage(m5);
			
			m5.clear();
			m5.setAddress("/hand/numtips");
			m5.addIntArg(tips.size());
			sender.sendMessage(m5);
			
			m6.setAddress("/hand/area");
			m6.addFloatArg(contourArea(Mat(approx)));
			sender.sendMessage(m6);
			
			
			vector<Point> newcontour;
			resampleVector(contours[j], newcontour, 40);
			
			m5.clear();
			m5.setAddress("/hand/dist");
			for (size_t i = 0; i < 40; i++) {
				float d = sqrt(newcontour[i].x*centerHullPoint.x + newcontour[i].y*centerHullPoint.y);
				m5.addFloatArg(d);
			}
			sender.sendMessage(m5);
												 
			//ofxOscBundle b;
			//b.addMessage(m);
			//b.addMessage(m2);
			//b.addMessage(m3);
			
			//sender.sendBundle(b);
												 
												 
				
			/*
			vector<float> xes, yes;
			int contour_size = contours[j].size();
			for (size_t i = 0; i < contour_size; i++) {
				xes.push_back(contours[j][i].x);
				yes.push_back(contours[j][i].y);
			}
			
			ofxOscMessage m5;
			m5.setAddress("/hand/x");
			float sampling = contour_size/20.0;
			vector<float> sampled_xes, sampled_yes;
			for (size_t i = 0; i < 20; i++) {
				float x = xes[floor(contour_size/(float)i)] / (float)height;
				sampled_xes.push_back(x);
				m5.addFloatArg(x);
			}
			sender.sendMessage(m);
			
			ofxOscMessage m6;
			m6.setAddress("/hand/y");
			for (size_t i = 0; i < 20; i++) {
				float y = yes[floor(contour_size/(float)i)] / (float)width;
				sampled_yes.push_back(y);
				m6.addFloatArg(y);
			}
			sender.sendMessage(m6);
			*/
			
		}
	}
}


// Handle creation of a new hand
static XnBool bShouldPrint = false;
void XnVPointDrawer::OnPointCreate(const XnVHandPointContext* cxt)
{
	printf("** %d\n", cxt->nID);
	// Create entry for the hand
	m_History[cxt->nID].clear();
	bShouldPrint = true;
	OnPointUpdate(cxt);
	bShouldPrint = true;
	
}
// Handle new position of an existing hand
void XnVPointDrawer::OnPointUpdate(const XnVHandPointContext* cxt)
{	
	// positions are kept in projective coordinates, since they are only used for drawing
	XnPoint3D ptProjective(cxt->ptPosition);
	ptProjective.X = prev_handpt.X*0.8 + ptProjective.X*0.2;
	ptProjective.Y = prev_handpt.Y*0.8 + ptProjective.Y*0.2;
	ptProjective.Z = prev_handpt.Z*0.8 + ptProjective.Z*0.2;
	prev_handpt = ptProjective;
	SegmentHand(ptProjective);
	
	imshow(hand_window_name, hand_matrix_img);
	
	if (bShouldPrint)printf("Point (%f,%f,%f)", ptProjective.X, ptProjective.Y, ptProjective.Z);
	m_DepthGenerator.ConvertRealWorldToProjective(1, &ptProjective, &ptProjective);
	if (bShouldPrint)printf(" -> (%f,%f,%f)\n", ptProjective.X, ptProjective.Y, ptProjective.Z);

	// Add new position to the history buffer
	m_History[cxt->nID].push_front(ptProjective);
	// Keep size of history buffer
	if (m_History[cxt->nID].size() > m_nHistorySize)
		m_History[cxt->nID].pop_back();
	bShouldPrint = false;
	
	
}

// Handle destruction of an existing hand
void XnVPointDrawer::OnPointDestroy(XnUInt32 nID)
{
	// No need for the history buffer
	m_History.erase(nID);
}

#define MAX_DEPTH 10000
float g_pDepthHist[MAX_DEPTH];
unsigned int getClosestPowerOfTwo(unsigned int n)
{
	unsigned int m = 2;
	while(m < n) m<<=1;

	return m;
}
GLuint initTexture(void** buf, int& width, int& height)
{
	GLuint texID = 0;
	glGenTextures(1,&texID);

	width = getClosestPowerOfTwo(width);
	height = getClosestPowerOfTwo(height); 
	*buf = new unsigned char[width*height*4];
	glBindTexture(GL_TEXTURE_2D,texID);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	return texID;
}

GLfloat texcoords[8];
void DrawRectangle(float topLeftX, float topLeftY, float bottomRightX, float bottomRightY)
{
	GLfloat verts[8] = {	topLeftX, topLeftY,
		topLeftX, bottomRightY,
		bottomRightX, bottomRightY,
		bottomRightX, topLeftY
	};
	glVertexPointer(2, GL_FLOAT, 0, verts);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	glFlush();
}
void DrawTexture(float topLeftX, float topLeftY, float bottomRightX, float bottomRightY)
{
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, texcoords);

	DrawRectangle(topLeftX, topLeftY, bottomRightX, bottomRightY);

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

void DrawDepthMap(const xn::DepthMetaData& dm)
{
	static bool bInitialized = false;	
	static GLuint depthTexID;
	static unsigned char* pDepthTexBuf;
	static int texWidth, texHeight;

	 float topLeftX;
	 float topLeftY;
	 float bottomRightY;
	 float bottomRightX;
	float texXpos;
	float texYpos;

	if(!bInitialized)
	{
		XnUInt16 nXRes = dm.XRes();
		XnUInt16 nYRes = dm.YRes();
		texWidth =  getClosestPowerOfTwo(nXRes);
		texHeight = getClosestPowerOfTwo(nYRes);

		depthTexID = initTexture((void**)&pDepthTexBuf,texWidth, texHeight) ;

		bInitialized = true;

		topLeftX = nXRes;
		topLeftY = 0;
		bottomRightY = nYRes;
		bottomRightX = 0;
		texXpos =(float)nXRes/texWidth;
		texYpos  =(float)nYRes/texHeight;

		memset(texcoords, 0, 8*sizeof(float));
		texcoords[0] = texXpos, texcoords[1] = texYpos, texcoords[2] = texXpos, texcoords[7] = texYpos;

	}
	unsigned int nValue = 0;
	unsigned int nHistValue = 0;
	unsigned int nIndex = 0;
	unsigned int nX = 0;
	unsigned int nY = 0;
	unsigned int nNumberOfPoints = 0;
	XnUInt16 g_nXRes = dm.XRes();
	XnUInt16 g_nYRes = dm.YRes();

	unsigned char* pDestImage = pDepthTexBuf;

	const XnUInt16* pDepth = dm.Data();

	// Calculate the accumulative histogram
	memset(g_pDepthHist, 0, MAX_DEPTH*sizeof(float));
	for (nY=0; nY<g_nYRes; nY++)
	{
		for (nX=0; nX<g_nXRes; nX++)
		{
			nValue = *pDepth;

			if (nValue != 0)
			{
				g_pDepthHist[nValue]++;
				nNumberOfPoints++;
			}

			pDepth++;
		}
	}

	for (nIndex=1; nIndex<MAX_DEPTH; nIndex++)
	{
		g_pDepthHist[nIndex] += g_pDepthHist[nIndex-1];
	}
	if (nNumberOfPoints)
	{
		for (nIndex=1; nIndex<MAX_DEPTH; nIndex++)
		{
			g_pDepthHist[nIndex] = (unsigned int)(256 * (1.0f - (g_pDepthHist[nIndex] / nNumberOfPoints)));
		}
	}

	pDepth = dm.Data();
	{
		XnUInt32 nIndex = 0;
		// Prepare the texture map
		for (nY=0; nY<g_nYRes; nY++)
		{
			for (nX=0; nX < g_nXRes; nX++, nIndex++)
			{
				nValue = *pDepth;

				if (nValue != 0)
				{
					nHistValue = g_pDepthHist[nValue];

					pDestImage[0] = nHistValue; 
					pDestImage[1] = nHistValue;
					pDestImage[2] = nHistValue;
				}
				else
				{
					pDestImage[0] = 0;
					pDestImage[1] = 0;
					pDestImage[2] = 0;
				}

				pDepth++;
				pDestImage+=3;
			}

			pDestImage += (texWidth - g_nXRes) *3;
		}
	}
	glBindTexture(GL_TEXTURE_2D, depthTexID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, pDepthTexBuf);

	// Display the OpenGL texture map
	glColor4f(0.5,0.5,0.5,1);

	glEnable(GL_TEXTURE_2D);
	DrawTexture(dm.XRes(),dm.YRes(),0,0);	
	glDisable(GL_TEXTURE_2D);
}

void glPrintString(void *font, char *str)
{
	int i,l = strlen(str);

	for(i=0; i<l; i++)
	{
		glutBitmapCharacter(font,*str++);
	}
}

void DrawFrameID(XnUInt32 nFrameID)
{
	glColor4f(1,0,0,1);
	glRasterPos2i(20, 50);
	XnChar strLabel[20];
	sprintf(strLabel, "%d", nFrameID);
	glPrintString(GLUT_BITMAP_HELVETICA_18, strLabel);
}

// Colors for the points
XnFloat Colors[][3] =
{
	{1,0,0},	// Red
	{0,1,0},	// Green
	{0,0.5,1},	// Light blue
	{1,1,0},	// Yellow
	{1,0.5,0},	// Orange
	{1,0,1},	// Purple
	{1,1,1}		// White. reserved for the primary point
};
XnUInt32 nColors = 6;

void XnVPointDrawer::Draw() const
{
	std::map<XnUInt32, std::list<XnPoint3D> >::const_iterator PointIterator;

	// Go over each existing hand
	for (PointIterator = m_History.begin();
		PointIterator != m_History.end();
		++PointIterator)
	{
		// Clear buffer
		XnUInt32 nPoints = 0;
		XnUInt32 i = 0;
		XnUInt32 Id = PointIterator->first;

		// Go over all previous positions of current hand
		std::list<XnPoint3D>::const_iterator PositionIterator;
		for (PositionIterator = PointIterator->second.begin();
			PositionIterator != PointIterator->second.end();
			++PositionIterator, ++i)
		{
			// Add position to buffer
			XnPoint3D pt(*PositionIterator);
			m_pfPositionBuffer[3*i] = pt.X;
			m_pfPositionBuffer[3*i + 1] = pt.Y;
			m_pfPositionBuffer[3*i + 2] = 0;//pt.Z();
		}
		
		// Set color
		XnUInt32 nColor = Id % nColors;
		XnUInt32 nSingle = GetPrimaryID();
		if (Id == GetPrimaryID())
			nColor = 6;
		// Draw buffer:
		glColor4f(Colors[nColor][0],
				Colors[nColor][1],
				Colors[nColor][2],
				1.0f);
		glPointSize(2);
		glVertexPointer(3, GL_FLOAT, 0, m_pfPositionBuffer);
		glDrawArrays(GL_LINE_STRIP, 0, i);

		glPointSize(8);
		glDrawArrays(GL_POINTS, 0, 1);
		glFlush();
	}
}

// Handle a new Message
void XnVPointDrawer::Update(XnVMessage* pMessage)
{
	// PointControl's Update calls all callbacks for each hand
	XnVPointControl::Update(pMessage);

	if (m_bDrawDM)
	{
		// Draw depth map
		xn::DepthMetaData depthMD;
		m_DepthGenerator.GetMetaData(depthMD);
		DrawDepthMap(depthMD);
	}
	if (m_bFrameID)
	{
		// Print out frame ID
		xn::DepthMetaData depthMD;
		m_DepthGenerator.GetMetaData(depthMD);
		DrawFrameID(depthMD.FrameID());
	}

	// Draw hands
	Draw();
}

void PrintSessionState(SessionState eState)
{
	glColor4f(1,0,1,1);
	glRasterPos2i(20, 20);
	XnChar strLabel[200];

	switch (eState)
	{
	case IN_SESSION:
		sprintf(strLabel, "Tracking hands"); break;
	case NOT_IN_SESSION:
		sprintf(strLabel, "Perform click or wave gestures to track hand"); break;
	case QUICK_REFOCUS:
		sprintf(strLabel, "Raise your hand for it to be identified, or perform click or wave gestures"); break;
	}

	glPrintString(GLUT_BITMAP_HELVETICA_18, strLabel);
}




//--------------------------------------------------------------------------------  
static int qsort_carea_compare( const void* _a, const void* _b) {  
    int out = 0;  
    // pointers, ugh.... sorry about this  
    CvSeq* a = *((CvSeq **)_a);  
    CvSeq* b = *((CvSeq **)_b);  
    // use opencv to calc size, then sort based on size  
    float areaa = fabs(cvContourArea(a, CV_WHOLE_SEQ));  
    float areab = fabs(cvContourArea(b, CV_WHOLE_SEQ));  
    // note, based on the -1 / 1 flip  
    // we sort biggest to smallest, not smallest to biggest  
    if( areaa > areab ) { out = -1; }  
    else {                out =  1; }  
    return out;  
}  



/*

//--------------------------------------------------------------------------------  
ofxCvConvexHull_hand::ofxCvConvexHull_hand() {  

	
}  

//--------------------------------------------------------------------------------  
ofxCvConvexHull_hand::~ofxCvConvexHull_hand() {  
    free( myMoments );  
}  

//--------------------------------------------------------------------------------  
void ofxCvConvexHull_hand::draw( float x, float y ) {  
	
    CvConvexityDefect* defectArrayDRAW;   
	
	
	
	if (hullsize>0)   
	{   
		
		ofSetColor(0xDD00CC);   
		glPushMatrix();   
		glTranslatef( x, y, 0.0 );   
		
		
        for(;defects;defects = defects->h_next)   
        {   
            int nomdef = defects->total; // defect amount   
            if(nomdef == 0)continue;   
            defectArrayDRAW = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*nomdef);   
            cvCvtSeqToArray(defects,defectArrayDRAW, CV_WHOLE_SEQ);   
			
            //draw defect array   
            ofSetColor(0x00FF00);   
			
            for(int i=0; i<nomdef; i++)   
            {   
				//ofBeginShape();   
				ofCircle(defectArrayDRAW[i].depth_point->x, defectArrayDRAW[i].depth_point->y,10);   
				//ofEndShape();   
            }   
            //glPopMatrix();   
			
            free(defectArrayDRAW);   
            if (defects)cvClearSeq(defects);   
			
        }   
	}  
}  

//--------------------------------------------------------------------------------  
int XnVPointDrawer::findPoints( IplImage *input) {  
    reset();   
	
    //START TO FIND THE HULL POINTS  
    cvFindContours( input, stor02, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );   
	
    if (contours)   
		contours = cvApproxPoly( contours, sizeof(CvContour), stor02, CV_POLY_APPROX_DP, 3, 1 );   
	
    int i = 0;   
    int area = 0;   
    int selected = -1;   
	
    CvSeq* first_contour;   
    first_contour = contours;   
	
    for( ; contours != 0; contours = contours->h_next )   
    {   
		CvRect rect;   
		int count = contours->total;   
		rect = cvContourBoundingRect(contours, 1);   
		if  ( (rect.width*rect.height) > area )   
		{   
			selected = i;   
			area = rect.width*rect.height;   
		}   
		i++;   
    }   
	
    contours = first_contour;   
	
    int k = 0;   
    for( ; contours != 0; contours = contours->h_next )   
	{   
        int i; // Indicator of cycles.   
        int count = contours->total; // This is number point in contour   
        CvPoint center;   
        CvSize size;   
        CvRect rect;   
		
        rect = cvContourBoundingRect( contours, 1);   
		
        if ( (k==selected) )   
        {   
			
            // Alloc memory for contour point set.   
            PointArray = (CvPoint*)malloc( count*sizeof(CvPoint) );   
			
            // Alloc memory for indices of convex hull vertices.   
            hull = (int*)malloc(sizeof(int)*count);   
			
            // Get contour point set.   
            cvCvtSeqToArray(contours, PointArray, CV_WHOLE_SEQ);   
			
            // Find convex hull for curent contour.   
            cvConvexHull( PointArray,   
						 count,   
						 NULL,   
						 CV_COUNTER_CLOCKWISE,   
						 hull,   
						 &hullsize);   
            // Find convex hull for current contour.   
            // This required for cvConvexityDefects().   
            seqhull = cvConvexHull2( contours,0,   
									CV_COUNTER_CLOCKWISE,   
									0);   
			
            // This required for cvConvexityDefects().   
            // Otherwise cvConvexityDefects() falled.   
            if( hullsize  4 )   
                continue;   
			
            // Find defects of convexity of current contours.   
            defects = cvConvexityDefects( contours,   
										 seqhull,   
										 stor03);   
			
            int nomdef = defects->total; // defect amount   
			
            // Free memory.   
            free(PointArray);   
            free(hull);   
			
        }   
        k++;   
    }   
	
	
    cvClearMemStorage( stor03 );   
    cvClearMemStorage( stor02 );   
    if (seqhull)cvClearSeq(seqhull);  
}
 
 */