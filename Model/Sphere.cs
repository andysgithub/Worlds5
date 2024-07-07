using System;
using System.Collections.Generic;
using System.Drawing;
using static Model.TracedRay;

namespace Model
{
    public class clsSphere
    {
        #region Private fields

        // Vertical field of view (degrees)                    
        private static float verticalView;
        // Horizontal field of view (degrees)
        private static float horizontalView;
        // The value to use when calculating surface patch angles
        private float incrementFactor;

        #endregion

        public clsSphere()
        {
            settings.PositionMatrix = new float[6, 6];
        }

        // Last generated image
        public Bitmap ViewportImage { get; set; }

        // Sinusoidal mapping of traced rays
        public TracedRay.RayDataType[,] RayMap { get; set; }

        public Settings settings;

        public struct Settings {

            #region Ray tracing properties

            // Transformation matrix
            public float[,] PositionMatrix { get; set; }

            // Sinusoidal mapping of surface points
            //public SurfacePoint[,] SurfaceMap { get; set; } 

            // The maximum number of points examined during ray tracing
            public int[] MaxSamples { get; set; }
            // The number of steps in the binary search for an orbit value boundary
            public int[] BinarySearchSteps { get; set; }
            // Bailout value for the fractal algorithm
            public float Bailout { get; set; }
            // Distance between sampling points during ray tracing
            public float[] SamplingInterval { get; set; }
            // The smoothing of the surface
            public float SurfaceSmoothing { get; set; }
            // The minimum acceptable thickness of the detected surface, to avoid speckling
            public float SurfaceThickness { get; set; }
            // The amount that the current orbit value is sufficiently different
            // from the last recorded sample to start a binary search for the boundary
            public float BoundaryInterval { get; set; }
            // Flag to indicate display of surface region / external region
            public int ActiveIndex { get; set; }
            // Flag to indicate whether to render in C++ or Cuda
            public bool CudaMode { get; set; }

            #endregion

            #region Rendering properties

            public float[] ExposureValue { get; set; }
            public float[] Saturation { get; set; }
            // The contrast of the surface shading
            public float SurfaceContrast { get; set; }
            // The lighting angle for surface shading (0 to 180 degrees)
            public float LightingAngle { get; set; }
            // The elevation angle from the xy plane for surface shading (0 to 90 degrees)
            public float LightElevationAngle { get; set; }

            #endregion

            #region Colour properties

            public float ColourCompression { get; set; }
            public float ColourOffset { get; set; }

            #endregion

            #region Transformed reference points

            // Top edge of the viewing window
            public float LatitudeStart { get; set; }
            // Bottom edge of the viewing window
            public float LatitudeEnd { get; set; }
            // Left edge of the viewing window as seen from the sphere centre
            public float LongitudeStart { get; set; }
            // Right edge of the viewing window
            public float LongitudeEnd { get; set; }

            #endregion

            #region Viewing window properties

            // Angular resolution of the viewport surface (degrees)
            public float AngularResolution { get; set; }
            // Latitude of the viewing centre (degrees)
            public float CentreLatitude { get; set; }
            // Longitude of the viewing centre (degrees)
            public float CentreLongitude { get; set; }
            // Distance from centre to first ray tracing point
            public float SphereRadius { get; set; }

            // Vertical field of view (degrees)
            public float VerticalView
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
            public float HorizontalView
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

            #region Clipping plane

            // Clipping plane rotation angles for each dimension
            public AxisPair ClippingAxes { get; set; }
            // Constant value for the clipping plane offset
            public float ClippingOffset { get; set; }
            // Flag to indicate if clipping should be used
            public bool UseClipping { get; set; }

            #endregion
        }

        public void InitialiseRayMap()
        {
            // Initialise the ray trace mapping to correspond to the viewport
            RayMap = new TracedRay.RayDataType[
                (int)(settings.HorizontalView / settings.AngularResolution) + 1,
                (int)(settings.VerticalView / settings.AngularResolution) + 1
            ];
            incrementFactor = 2 * (float)Math.Sin(settings.AngularResolution / 2);
        }

        public void RecordRay(TracedRay tracedRay, int xIndex, int yIndex)
        {
            // Store the new ray in the ray map
            RayMap[xIndex, yIndex] = tracedRay.RayData;
        }

        public List<float> addTiltValues(TracedRay tracedRay, RenderingParams renderParams, int xIndex, int yIndex)
        {
            // Initialise the tilt values list
            List<float> tiltValues = new List<float>();
            RayDataType rayData = RayMap[xIndex, yIndex];

            TracedRay lastRay = new TracedRay(rayData.ExternalPoints, rayData.ModulusValues, rayData.AngleValues, rayData.DistanceValues, renderParams);

            if (lastRay != null)
            {
                int lastRayStart = 0;

                // For each surface boundary distance in the current ray
                for (int pointCount = 0; pointCount < tracedRay.BoundaryTotal && !float.IsPositiveInfinity(tracedRay.Boundary(pointCount)); pointCount++)
                {
                    float tiltValue = 0;

                    // If this is a surface point
                    if (tracedRay.IsSurfacePoint(pointCount))
                    {
                        // Get the distance to the surface
                        float currentDistance = tracedRay.Boundary(pointCount);
                        float increment = currentDistance * incrementFactor;

                        for (int testCount = lastRayStart; testCount < lastRay.BoundaryTotal && !float.IsPositiveInfinity(lastRay.Boundary(testCount)); testCount++, lastRayStart++)
                        {
                            // If this is a surface point
                            if (lastRay.IsSurfacePoint(testCount))
                            {
                                // Get the distance to the surface
                                float lastDistance = lastRay.RayData.DistanceValues[testCount];

/*                                // If the distance on the previous ray has overshot the current ray
                                if (lastDistance > currentDistance + settings.SurfaceThickness * 10.0)
                                {
                                    // Go on to the next current ray surface distance
                                    break;
                                }

                                // If this is not an adjacent surface point
                                if (lastDistance < currentDistance - settings.SurfaceThickness * 10)
                                {
                                    // Go on to the next test ray surface distance
                                    continue;
                                }*/

                                // Calculate the longitudinal tilt value
                                float separation = lastDistance - currentDistance;
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
    }
}
