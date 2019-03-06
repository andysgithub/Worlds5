using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.IO;
using System.Threading;
using Model;

namespace Worlds5 
{
	sealed public class ImageRendering
	{
        #region Member Variables

        //  Image display settings
		private static Size m_ImageSize;		// Width & Height of image from file
        private static Size m_SeqSize;			// Width & Height of sequence frames
        private static double m_ScaleValue;	    // Overall scaling value for matrix
        
		//  Image quality settings
        private static float m_Bailout; 
        
		//  Sequence playback
        private static int m_CurrentKey;		// Current key frame for sequence
        private static int m_FrameCount;		// Frame to display for current key

        #endregion

        #region Class Properties

        public static Size ImageSize
        {
            get { return m_ImageSize; }
            set { m_ImageSize = value; }
        }

        public static Size SeqSize
        {
            get { return m_SeqSize; }
            set { m_SeqSize = value; }
        }

        public static double ScaleValue
        {
            get { return m_ScaleValue; }
            set { m_ScaleValue = value; }
        }

        public static float Bailout
        {
            get { return m_Bailout; }
            set { m_Bailout = value; }
        }

        public static int CurrentKey
        {
            get { return m_CurrentKey; }
            set { m_CurrentKey = value; }
        }

        public static int FrameCount
        {
            get { return m_FrameCount; }
            set { m_FrameCount = value; }
        }

        #endregion
	}

    public class RenderThread
    {
        private BackgroundWorker m_bwThread;
        private clsSphere sphere;

        [DllImport("Unmanaged.dll")]
        static extern void TraceRay(double startDistance, double increment, double surfaceThickness,
            double xFactor, double yFactor, double zFactor,
            [Out, MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] bool[] externalsArray,
            float[] valuesArray, float[] anglesArray, double[] distancesArray,
            int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
            bool showSurface, bool showExterior);

        //[DllImport("Unmanaged.dll")]
        //static extern void TraceSurface(double startDistance, double increment, double surfaceThickness, 
        //                                double XFactor, double YFactor, double ZFactor,
        //                                ref float modulus, ref float angle, ref double distance,
        //                                int rayPoints, int maxSamples, int binarySearchSteps);

        [DllImport("Unmanaged.dll")]
        static extern void InitSphere(float fDetail0, float fDetail1,
                                          float fBailout, double Resolution,
                                          float fHue0, float fSaturation0, float fLightness0,
                                          float fHue1, float fSaturation1, float fLightness1,
                                          double Latitude, double Longitude,
                                          double Radius, double verticalView, double horizontalView, double[,] PositionMatrix);

        public RenderThread(float Bailout, BackgroundWorker bwThread)
        {
            sphere = Model.Globals.Sphere;
            m_bwThread = bwThread;

            InitSphere(sphere.ColourDetail[0], sphere.ColourDetail[1],
                       Bailout, sphere.AngularResolution,
                       sphere.HSL[0, 0], sphere.HSL[1, 0], sphere.HSL[2, 0],
                       sphere.HSL[0, 1], sphere.HSL[1, 1], sphere.HSL[2, 1],
                       sphere.CentreLatitude, sphere.CentreLongitude,
                       sphere.Radius, sphere.VerticalView, sphere.HorizontalView, sphere.PositionMatrix);
        }

        public void RenderRays(int rayCountY)
        {
            clsSphere sphere = Model.Globals.Sphere;
            double latitude = sphere.LatitudeStart - rayCountY * sphere.AngularResolution;

            // Process this line of latitude
            ProcessLatitudeLine(rayCountY, latitude);

            if (m_bwThread.CancellationPending)
            {
                return;
            }

            // Extend the boundaries across all rays in this row
            //extendBoundaries(rayCountY);

            displayLine(sphere, rayCountY, latitude);
        }

        //bool redisplay = false;
        public void Redisplay(int rayCountY)
        {
            clsSphere sphere = Model.Globals.Sphere;
            double latitude = sphere.LatitudeStart - rayCountY * sphere.AngularResolution;

            if (m_bwThread.CancellationPending)
            {
                return;
            }
            //redisplay = true;
            // Display this line of latitude
            displayLine(sphere, rayCountY, latitude);
        }

        // Trace each ray for this latitude line and store in the sphere
        private void ProcessLatitudeLine(int rayCountY, double latitude)
        {
            int rayCountX = 0;

            // For each longitude point on this line
            for (double longitude = sphere.LongitudeStart;
                longitude > sphere.LongitudeEnd; longitude -= sphere.AngularResolution)
            {
                if (m_bwThread.CancellationPending)
                {
                    return;
                }

                double xFactor = Math.Cos(latitude * Globals.DEG_TO_RAD) * Math.Sin(-longitude * Globals.DEG_TO_RAD);
                double yFactor = Math.Sin(latitude * Globals.DEG_TO_RAD);
                double zFactor = Math.Cos(latitude * Globals.DEG_TO_RAD) * Math.Cos(-longitude * Globals.DEG_TO_RAD);

                TracedRay tracedRay;
                bool[] externalPoints = new bool[100];
                float[] modulusValues = new float[100];
                float[] angleValues = new float[100];
                double[] distanceValues = new double[100];

                // Trace the ray from the sphere radius outwards
                TraceRay(sphere.Radius, sphere.SamplingInterval, sphere.SurfaceThickness,
                            xFactor, yFactor, zFactor,
                            externalPoints, modulusValues, angleValues, distanceValues,
                            sphere.RayPoints, sphere.MaxSamples, sphere.BoundaryInterval, sphere.BinarySearchSteps,
                            sphere.ShowSurface, sphere.ShowExterior);

                // Record the fractal value collection for this ray
                tracedRay = new TracedRay(externalPoints, modulusValues, angleValues, distanceValues);

                if (tracedRay.ModulusValues[1] != 0)
                {
                    var x = 0;
                }

                // Add this ray to the ray map in the sphere
                sphere.addRay(tracedRay, rayCountX, rayCountY);

                rayCountX++;
            }
        }

        private void displayLine(clsSphere sphere, int rayCountY, double latitude)
        {
            int rayCountX = 0;

            // For each longitude point on this line
            for (double longitude = sphere.LongitudeStart;
                longitude > sphere.LongitudeEnd; longitude -= sphere.AngularResolution)
            {
                if (rayCountX >= sphere.RayMap.GetUpperBound(0))
                {
                    break;
                }
                // Get the ray from the ray map
                TracedRay tracedRay = sphere.RayMap[rayCountX++, rayCountY];

                // Calculate the tilt values from the previous rays
                if (rayCountX > 0)
                {
                    tracedRay.XTiltValues = sphere.addTiltValues(tracedRay, rayCountX - 1, rayCountY);
                }
                if (rayCountY > 0)
                {
                    tracedRay.YTiltValues = sphere.addTiltValues(tracedRay, rayCountX, rayCountY - 1);
                }

                if (tracedRay != null)
                {
                    // Convert the fractal value collection into an rgb colour value
                    tracedRay.SetColour(sphere.ExposureValue, sphere.Saturation, sphere.StartDistance, sphere.EndDistance);
                    m_bwThread.ReportProgress(-1, new object[] { latitude, longitude, tracedRay });
                }
            }
            m_bwThread.ReportProgress(rayCountY);
        }

        //private void extendBoundaries(int row)
        //{
        //    // Start with the first ray
        //    int rayNumber = 0;

        //    // For each longitude point on this line working forwards
        //    for (double longitude = sphere.LongitudeStart - sphere.Resolution;
        //        longitude > sphere.LongitudeEnd; longitude -= sphere.Resolution)
        //    {
        //        // Get the ray from the ray map
        //        TracedRay thisRay = sphere.RayMap[rayNumber, row];
        //        // Get the next ray from the ray map
        //        TracedRay nextRay = sphere.RayMap[rayNumber + 1, row];
        //        // Add any additional boundaries on this ray to the next one
        //        addNewBoundaries(thisRay, ref nextRay);
        //        // Go on to the next ray in the row
        //        rayNumber++;
        //    }

        //    // Start with the last ray

        //    // For each longitude point on this line working backwards
        //    for (double longitude = sphere.LongitudeEnd + sphere.Resolution;
        //        longitude < sphere.LongitudeStart; longitude += sphere.Resolution)
        //    {
        //        // Get the ray from the ray map
        //        TracedRay thisRay = sphere.RayMap[rayNumber, row];
        //        // Get the previous ray from the ray map
        //        TracedRay previousRay = sphere.RayMap[rayNumber - 1, row];
        //        // Add any additional boundaries on this ray to the previous one
        //        addNewBoundaries(thisRay, ref previousRay);
        //        // Go on to the previous ray in the row
        //        rayNumber--;
        //    }
        //}

        ///// <summary>
        ///// Find extra boundaries from the source list which aren't in the target list
        ///// and return a new target list.
        ///// </summary>
        ///// <param name="sourceBoundaries">The list of boundary distances to test.</param>
        ///// <param name="targetBoundaries">The list of boundary distances to add to.</param>
        ///// <returns></returns>
        //private void addNewBoundaries(TracedRay sourceRay, ref TracedRay targetRay)
        //{
        //    List<double> newBoundaries = targetRay.Boundaries;
        //    int sourcePos = -1;
        //    int targetPos = 0;

        //    // For each source boundary position value
        //    foreach (double sourcePosition in sourceRay.Boundaries)
        //    {
        //        sourcePos++;
        //        bool boundaryFound = false;

        //        // Search for the nearest item in the target list
        //        while (targetRay.Boundary(targetPos) < sourcePosition)
        //        {
        //            if (targetRay.Boundary(targetPos + 1) > sourcePosition)
        //            {
        //                boundaryFound = true;
        //                break;
        //            }
        //            targetPos++;
        //        }

        //        if (Double.IsPositiveInfinity(targetRay.Boundary(targetPos)))
        //        {
        //            return;
        //        }

        //        // If no position found to insert this boundary
        //        if (!boundaryFound)
        //        {
        //            // Go on to the next source boundary position
        //            continue;
        //        }

        //        // If this angle is sufficiently close to the test angle
        //        if (Math.Abs(sourceRay.Angle(sourcePos) - targetRay.Angle(targetPos)) < sphere.BoundaryInterval)
        //        {
        //            // Go on to the next source boundary position
        //            continue;
        //        }

        //        // Set increment to the distance between the target and source points
        //        double separation = sourcePosition - targetRay.Boundary(targetPos);

        //        float Modulus, Angle;
        //        bool externalPoint;

        //        // Perform binary search between the target and the source points, to determine new boundary position
        //        double newPosition = FindBoundary(separation, sphere.BinarySearchSteps, sourcePosition, targetRay.Angle(targetPos),
        //                              sphere.BoundaryInterval, &externalPoint, &Modulus, &Angle);

        //        // Insert this new boundary data into the current position in the target ray
        //        targetRay.Boundaries.Insert(targetPos+1, newPosition);
        //        targetRay.AngleValues.Insert(targetPos+1, Angle);
        //        targetRay.ModulusValues.Insert(targetPos+1, Modulus);
        //        targetRay.ExternalPoints.Insert(targetPos+1, externalPoint);
        //        targetPos++;
        //    }
        //}

    }
}
