/******************************************************************************
 * Hough line detection
 *
 * Author: Cristina Grama
 * Updated: 13.08.2015 
 *****************************************************************************/

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <math.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <time.h>

using namespace cv;

// Conversion factor from degrees to radians
const double conv = M_PI / 180;

struct colour {
   int blue;
   int green;
   int red;
} colours[7];

/*
   Generates a number of 7 colours that can be used for 
   drawing the detected lines.
   The colours are stored as an opencv Scalar(B, G, R)
 */
void generateColours()
{
   colours[0].blue = 0;
   colours[0].green = 0;
   colours[0].red = 255;
   colours[1].blue = 255;
   colours[1].green = 0;
   colours[1].red = 0;
   colours[2].blue = 0;
   colours[2].green = 255;
   colours[2].red = 0;
   colours[3].blue = 255;
   colours[3].green = 255;
   colours[3].red = 0;
   colours[4].blue = 255;
   colours[4].green = 0;
   colours[4].red = 255;
   colours[5].blue = 0;
   colours[5].green = 255;
   colours[5].red = 255;
   colours[6].blue = 127;
   colours[6].green = 127;
   colours[6].red = 127;
}

/*
   Parameters:
   img - the original image
   nmax - the number of maxima to be searched for
   
   r0 - array holding the r values of the maxima
   omega0 - array holding the omega values of the maxima
 */
void houghtransform(Mat img, int nmax, double *r0, double *omega0)
{
   // Define the origin for the Hough transform computation
   // In this case, it's the centre of the image
   int yc = img.rows / 2;
   int xc = img.cols / 2;
   
   // Define rmax as half of the image's diagonal:
   int rmax = sqrt(img.rows * img.rows + img.cols * img.cols) / 2;
      
   // Define the matrix which will hold the Hough image
   // It has the omega angles on the rows, ranging between 0 and 180
   // (the actual number of values is defined by the granularity)
   // And lengths on the columns, ranging from 0 to 2 * rmax
   // We make it bigger because we need a 1 pixel margin to make max search easier
   int z[182][2 * rmax + 3];
   // And initialise it with zeroes:
   for (int omega = 0; omega < 182; omega++)
      for (int r = 0; r < 2 * rmax + 3; r++)
         z[omega][r] = 0;
   
   // Build a lookup table of sin and cos values between 0 and 180
   // with a 1 degree granularity:
   double sin_lut[180];
   double cos_lut[180];
   for (int omega = 0; omega < 180; omega++)
   {
      sin_lut[omega] = sin(omega);
      cos_lut[omega] = cos(omega);
   }
      
   /* 
      Step 1: for each pixel in the original image, compute for every 
      possible omega angle step the line in Hough space, using the 
      line's normal form
   */
   int ri;
   for (int y = 0; y < img.rows; y++)
      for (int x = 0; x < img.cols; x++)
         if (img.at<uchar>(y, x) == 0) // only if the pixel is set
            for (int omega = 0; omega < 180; omega++)
            {
               // We shift the pixel w.r.t. the origin (xc, yc)
//               ri = (i - xc) * cos_lut[omega] + (j - yc) * sin_lut[omega];
               ri = (x - xc) * cos(omega * conv) + (y - yc) * sin(omega * conv);
               // And we increment the appropriate entry in the Hough space matrix z:
               z[omega + 1][rmax + ri + 1] += 1;
               // The +1s are necessary because of the 1 pixel margin
            }

   /*
      Step 2: find the maxima in the Hough space; these correspond to lines
      in the original image
      
      We find the maximum by searching for weighted maxima in a 3x3 neighbourhood      
   */
   double zmax = -1.0, crtmax = 0.0;
   int crtr, crtomega;
   for (int i = 0; i < nmax; i++)
   {
      zmax = -1.0;
      crtmax = 0.0;
      for (int omega = 1; omega < 181; omega++)
         for (int r = 1; r < 2 * rmax + 2; r++)
         {
            crtmax = (z[omega-1][r-1] + z[omega-1][r] + z[omega-1][r+1] + z[omega][r-1] + z[omega][r] + z[omega][r+1] + z[omega+1][r-1] + z[omega+1][r] + z[omega+1][r+1]) / 9;
            
            if (crtmax > zmax)
            {
               zmax = crtmax;
               crtr = r - 1; 
               crtomega = omega - 1;
               // -1s because we need to take into account the fake 1 pixel margin
            }
         }
      printf("Maximum #%d is: zmax=%f | r=%i | omega=%i \n", i, zmax, crtr - rmax, crtomega);
      if (zmax != 0)
      {
         r0[i] = (double)(z[crtomega][crtr-1] * (crtr - 1) + z[crtomega][crtr] * crtr + z[crtomega][crtr+1] * (crtr + 1)) / (double)(z[crtomega][crtr-1] + z[crtomega][crtr] + z[crtomega][crtr+1]);
         omega0[i] = (double)(z[crtomega-1][crtr] * (crtomega - 1) + z[crtomega][crtr] * crtomega + z[crtomega+1][crtr] * (crtomega + 1)) / (double)(z[crtomega-1][crtr] + z[crtomega][crtr] + z[crtomega+1][crtr]);
      }
      else // take the centre
      {
         omega0[i] = crtomega;
         r0[i] = crtr;
      }
      
      // And delete the rmax that we added in step 1 when doing z[omega][rmax + ri] += 1 !!!
      r0[i] = r0[i] - (double)rmax;
      
      // Also set to zero the neighbourhood for the found max, so we can find the next one:
      z[crtomega-1][crtr-1] = 0;
      z[crtomega-1][crtr] = 0;
      z[crtomega-1][crtr+1] = 0;
      z[crtomega][crtr-1] = 0;
      z[crtomega][crtr] = 0;
      z[crtomega][crtr+1] = 0;
      z[crtomega+1][crtr-1] = 0;
      z[crtomega+1][crtr] = 0;
      z[crtomega+1][crtr+1] = 0;
   }
      /*
         Make an opencv image out of the Hough image, for visualisation
         We need to scale the values in the z matrix so that they are between 0 and 255
      */
      int max = 0;
      for (int omega = 1; omega < 181; omega++)
         for (int r = 1; r < 2 * rmax + 2; r++)
            if (z[omega][r] > max) max = z[omega][r];
            
      Mat houghImage(180, 2 * rmax + 1, CV_8U);   
      for (int omega = 1; omega < 181; omega++)
         for (int r = 1; r < 2 * rmax + 2; r++)
            houghImage.at<uchar>(omega - 1, r - 1) = (uchar) (z[omega][r] * 255 / max);   
      
      // Do some magic to transform the i into a string
      std::string imgName;
      std::stringstream out;
      out << 4;
      imgName = out.str();
      // so we can have dynamically named images, so we can show the hough space at each iteration :-)
      imshow(imgName, houghImage);      
}

int main(int argc, char **argv)
{
   /**************************************
      IMAGE INPUT & PRE-PROCESSING STEPS
    *************************************/
   char *imageName = argv[1];
   Mat srcImage, grayImage;
   srcImage = imread(imageName, 1);
   if (argc != 2 || !srcImage.data)
   {
      printf("No image data! \n ");
      return -1;
   }

   // Convert to a grayscale image:
   cvtColor(srcImage, grayImage, CV_BGR2GRAY, CV_8U);
   
   // And binarise it:
   for (int i = 0; i < grayImage.rows; i++)
      for (int j = 0; j < grayImage.cols; j++)
         if (grayImage.at<uchar>(i, j) >= 127)
            grayImage.at<uchar>(i, j) = 255;
         else
            grayImage.at<uchar>(i, j) = 0;            

   /*************************
      HOUGH TRANSFORM STEPS
    ************************/
   int nmax = 3;
   double r0[nmax];
   double omega0[nmax];
   houghtransform(grayImage, nmax, r0, omega0);
   
   /*******************************
      DETECTED LINE DRAWING STEPS
    ******************************/   
   generateColours();
   srand(time(NULL));
   int colIndex;
    
   for (int k = 0; k < nmax; k++)
   {
      int i = k;
      printf("Drawing maximum #%d: r0=%f, omega0=%f\n", i, r0[i], omega0[i]);
      // Compute the (x, y) value pairs for the detected line
      // x are columns, y are rows! (cf. math notation)
      
      // We reformat the normal line equation to a y = mx + b form:
      double m = 0.0;
      if (sin(omega0[i] * conv) != 0)
         m = (double)(-cos(omega0[i] * conv)) / (double)(sin(omega0[i] * conv));
      else
      {
         printf("sin(omega0) = 0!\n");      
         m = 1000000; // "infinity"
      }
      
      colIndex = rand() % 7;
      if ((m >= -1) && (m <= 1))
      {
//         printf("The slope of the detected line is between [-1, 1]...\n");
         int xvals[grayImage.cols]; 
         int yvals[grayImage.cols];
         for (int x = 0; x < grayImage.cols; x++)
         {
            xvals[x] = x;
            yvals[x] = round((-cos(omega0[i] * conv) / sin(omega0[i] * conv)) * (double)(x - (grayImage.cols / 2)) + (r0[i] / sin(omega0[i] * conv)) + (double)(grayImage.rows / 2));
         }
         printf("(xval0 = %d, yval0 = %d) to (xvaln = %d, yvaln = %d) \n", xvals[0], yvals[0], xvals[grayImage.cols - 1], yvals[grayImage.cols - 1]);
         line(srcImage, Point(xvals[0], yvals[0]), Point(xvals[grayImage.cols - 1], yvals[grayImage.cols - 1]), Scalar(colours[colIndex].blue, colours[colIndex].green, colours[colIndex].red), 2, 8);
      }
      else
      {
//         printf("The slope of the detected line is in (-inf, -1) U (1, +inf)...\n");
         int yvals[grayImage.rows]; 
         int xvals[grayImage.rows];
         for (int y = 0; y < grayImage.rows; y++)
         {
            yvals[y] = y;
            xvals[y] = round((-sin(omega0[i] * conv) / cos(omega0[i] * conv)) * (double)(y - (grayImage.rows / 2)) + (r0[i] / cos(omega0[i] * conv)) + (double)(grayImage.cols / 2));
         }
         printf("(xval0 = %d, yval0 = %d) to (xvaln = %d, yvaln = %d) \n", xvals[0], yvals[0], xvals[grayImage.rows - 1], yvals[grayImage.rows - 1]);
         line(srcImage, Point(xvals[0], yvals[0]), Point(xvals[grayImage.rows - 1], yvals[grayImage.rows - 1]), Scalar(colours[colIndex].blue, colours[colIndex].green, colours[colIndex].red), 2, 8);
      }
      imshow("Detected lines", srcImage);
   }      
   waitKey(0);

   return 0;
}
