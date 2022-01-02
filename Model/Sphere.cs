using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace Model
{
    public class clsSphere
    {
        #region Private fields

        // Transformation matrix
        public double[,] PositionMatrix = new double[6, 6];
        // Vertical field of view (degrees)                    
        private double verticalView;
        // Horizontal field of view (degrees)
        private double horizontalView;
        // The value to use when calculating surface patch angles
        private double incrementFactor;                 

        // Image settings
        public float[] ColourDetail = new float[2];

        #endregion

        public clsSphere()
        {
        }

        #region Ray tracing properties

        // Sinusoidal mapping of traced rays
        public TracedRay.RayDataType[,] RayMap { get; set; }
        // Sinusoidal mapping of surface points
        //public SurfacePoint[,] SurfaceMap { get; set; } 
        // The number of boundary points recorded during ray tracing
        public int[] RayPoints { get; set; }
        // The maximum number of points examined during ray tracing
        public int[] MaxSamples { get; set; }
        // The number of steps in the binary search for an orbit value boundary
        public int[] BinarySearchSteps { get; set; }
        // Bailout value for the fractal algorithm
        public float Bailout { get; set; }
        // Distance between sampling points during ray tracing
        public double[] SamplingInterval { get; set; }
        // The minimum acceptable thickness of the detected surface, to avoid speckling
        public double SurfaceThickness { get; set; }
        // The amount that the current orbit value is sufficiently different
        // from the last recorded sample to start a binary search for the boundary
        public double BoundaryInterval { get; set; }
        // Flag to indicate display of surface region / external region
        public int ActiveIndex { get; set; }
        // Last generated image
        public Bitmap ViewportImage { get; set; }

        #endregion

        #region Rendering properties

        public float[] ExposureValue { get; set; }
        public float[] Saturation { get; set; }
        // The contrast of the surface shading
        public float SurfaceContrast { get; set; }
        // The lighting angle for surface shading (0 to 180 degrees)
        public float LightingAngle { get; set; }
        // The distance along the ray to start rendering the image
        public double[] StartDistance { get; set; }
        // The distance along the ray to finish rendering the image
        public double[] EndDistance { get; set; }
        public float InteriorExposure { get; set; }
        public float InteriorSaturation { get; set; }


        #endregion

        #region Transformed reference points

        // Top edge of the viewing window
        public double LatitudeStart { get; set; }
        // Bottom edge of the viewing window
        public double LatitudeEnd { get; set; }
        // Left edge of the viewing window as seen from the sphere centre
        public double LongitudeStart { get; set; }
        // Right edge of the viewing window
        public double LongitudeEnd { get; set; }        

        #endregion

        #region Viewing window properties

        // Angular resolution of the viewport surface (degrees)
        public double AngularResolution { get; set; }
        // Latitude of the viewing centre (degrees)
        public double CentreLatitude { get; set; }
        // Longitude of the viewing centre (degrees)
        public double CentreLongitude { get; set; }
        // Distance from centre to first ray tracing point
        public double Radius { get; set; }              

        // Vertical field of view (degrees)
        public double VerticalView                      
        {
            get { return verticalView; }
            set
            {
                verticalView = value;
                // Determine viewport edge positions
                LatitudeStart = value / 2;
                LatitudeEnd = -value / 2;
            }
        }

        // Horizontal field of view (degrees)
        public double HorizontalView                    
        {
            get { return horizontalView; }
            set
            {
                horizontalView = value;
                // Determine viewport edge positions
                LongitudeStart = value / 2;
                LongitudeEnd = -value / 2;
            }
        }

        #endregion

        //public void InitialiseSurfaceMap()
        //{
        //    // Initialise the surface mapping to correspond to the viewport
        //    surfaceMap = new SurfacePoint[(int)(HorizontalView / AngularResolution) + 1, (int)(VerticalView / AngularResolution) + 1];
        //    incrementFactor = 2 * Math.Sin(AngularResolution / 2);
        //}

        public void InitialiseRayMap()
        {
            // Initialise the ray trace mapping to correspond to the viewport
            RayMap = new TracedRay.RayDataType[(int)(HorizontalView / AngularResolution) + 1, (int)(VerticalView / AngularResolution) + 1];
            incrementFactor = 2 * Math.Sin(AngularResolution / 2);
        }

        public void RecordRay(TracedRay tracedRay, int xIndex, int yIndex)
        {
            // Store the new ray in the ray map
            RayMap[xIndex, yIndex] = tracedRay.RayData;
        }

        public List<float> addTiltValues(TracedRay tracedRay, int xIndex, int yIndex)
        {
            // Initialise the tilt values list
            List<float> tiltValues = new List<float>();
            TracedRay.RayDataType rayData = RayMap[xIndex, yIndex];
            TracedRay lastRay = new TracedRay(rayData.ExternalPoints, rayData.ModulusValues, rayData.AngleValues, rayData.DistanceValues);
            if (lastRay != null)
            {
                int lastRayStart = 0;

                // For each surface boundary distance in the current ray
                for (int pointCount = 0; pointCount < tracedRay.BoundaryTotal && !double.IsPositiveInfinity(tracedRay.Boundary(pointCount)); pointCount++)
                {
                    float tiltValue = 0;

                    // If this is a surface point
                    if (tracedRay.IsSurfacePoint(pointCount))
                    {
                        // Get the distance to the surface
                        double currentDistance = tracedRay.Boundary(pointCount);
                        double increment = currentDistance * incrementFactor;

                        for (int testCount = lastRayStart; testCount < lastRay.BoundaryTotal && !double.IsPositiveInfinity(lastRay.Boundary(testCount)); testCount++, lastRayStart++)
                        {
                            // If this is a surface point
                            if (lastRay.IsSurfacePoint(testCount))
                            {
                                // Get the distance to the surface
                                double lastDistance = lastRay.RayData.DistanceValues[testCount];

                                // If the distance on the previous ray has overshot the current ray
                                if (lastDistance > currentDistance + SurfaceThickness * 10.0)
                                {
                                    // Go on to the next current ray surface distance
                                    break;
                                }

                                // If this is not an adjacent surface point
                                if (lastDistance < currentDistance - SurfaceThickness * 10)
                                {
                                    // Go on to the next test ray surface distance
                                    continue;
                                }

                                // Calculate the longitudinal tilt value
                                double separation = lastDistance - currentDistance;
                                tiltValue = (float)Math.Atan(separation / increment);
                                lastRayStart++;
                                break;
                            }
                        }
                    }
                    tiltValues.Add(tiltValue);
                }
            }
            return tiltValues;
        }

        //public void addSurfacePoint(SurfacePoint surfacePoint, int xIndex, int yIndex)
        //{
        //    surfaceMap[xIndex, yIndex] = surfacePoint;

        //    double increment = surfacePoint.Distance * incrementFactor;
        //    double lastDistance;
        //    double separation;

        //    if (xIndex > 0)
        //    {
        //        // Calculate the longitudinal tilt value from the previous longitude point
        //        lastDistance = surfaceMap[xIndex - 1, yIndex].Distance;
        //        separation = lastDistance - surfacePoint.Distance;
        //        surfaceMap[xIndex, yIndex].XTilt = (float)(Math.Atan(separation / increment));
        //    }

        //    if (yIndex > 0)
        //    {
        //        // Calculate the latitudinal tilt value from the previous latitude point
        //        lastDistance = surfaceMap[xIndex, yIndex - 1].Distance;
        //        separation = lastDistance - surfacePoint.Distance;
        //        surfaceMap[xIndex, yIndex].YTilt = (float)(Math.Atan(separation / increment));
        //    } 
        //}

    }
}
