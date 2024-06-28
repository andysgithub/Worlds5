using System;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Forms;
using Model;
using Newtonsoft.Json;
using static Worlds5.RayProcessing;

namespace Worlds5
{
    sealed public class RayProcessing
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct RayTracingParams
        {
            public float startDistance;
            public float increment;
            public float smoothness;
            public float surfaceThickness;
            public float bailout;
            public int rayPoints;
            public int maxSamples;
            public float boundaryInterval;
            public int binarySearchSteps;
            public int activeIndex;
        }

        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern int TraceRay(float startDistance, float increment, float smoothness, float surfaceThickness,
            float xFactor, float yFactor, float zFactor, float bailout,
            int[] externalsArray, float[] valuesArray, float[] anglesArray, float[] distancesArray,
            int rayPoints, int maxSamples, float boundaryInterval, int binarySearchSteps,
            int activeIndex);

        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern bool InitializeGPU(ref RayTracingParams parameters);

        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern bool CopyTransformationMatrix([In] float[] positionMatrix);

        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern bool VerifyTransformationMatrix([Out] float[] output);

        //[DllImport("Unmanaged.dll")]
        //static extern float[] ImageToFractalSpace (float startDistance, float xFactor, float yFactor, float zFactor);

        private int[] externalPoints;
        private float[] modulusValues;
        private float[] angleValues;
        private float[] distanceValues;

        public RayProcessing()
        {
            externalPoints = new int[100];
            modulusValues = new float[100];
            angleValues = new float[100];
            distanceValues = new float[100];
        }

        // Trace the ray on this latitude line
        public void ProcessRay(clsSphere sphere, int rayCountX, int rayCountY)
        {
            clsSphere.Settings settings = sphere.settings;
            float latitude = settings.LatitudeStart - rayCountY * settings.AngularResolution;
            float longitude = settings.LongitudeStart - rayCountX * settings.AngularResolution;
            int i = settings.ActiveIndex;
            int rayPoints = (int)(settings.MaxSamples[i] * settings.SamplingInterval[i]);

            float latRadians = latitude * Globals.DEG_TO_RAD;
            float longRadians = longitude * Globals.DEG_TO_RAD;

            float xFactor = (float)Math.Cos(latRadians) * (float)Math.Sin(-longRadians);
            float yFactor = (float)Math.Sin(latRadians);
            float zFactor = (float)Math.Cos(latRadians) * (float)Math.Cos(-longRadians);

            // Set the start distance to the sphere radius
            float startDistance = settings.Radius;

            // If clipping is enabled
            if (settings.UseClipping)
            {
                // Get the 5D coordinates for the intersection between this vector and the clipping plane
                float distance = Clipping.CalculateDistance(latRadians, longRadians, settings.ClippingAxes, settings.ClippingOffset);

                // Set the start distance to this value if larger than sphere radius
                if (distance > startDistance) startDistance = distance;
            }         

            // Trace the ray from the starting point outwards
            int points = TraceRay(startDistance, settings.SamplingInterval[i], settings.SurfaceSmoothing, settings.SurfaceThickness,
                        xFactor, yFactor, zFactor, settings.Bailout,
                        externalPoints, modulusValues, angleValues, distanceValues,
                        rayPoints, settings.MaxSamples[i], settings.BoundaryInterval, settings.BinarySearchSteps[i],
                        i);

            // Resize arrays to the recordedPoints value
            Array.Resize(ref externalPoints, points);
            Array.Resize(ref modulusValues, points);
            Array.Resize(ref angleValues, points);
            Array.Resize(ref distanceValues, points);

            Console.Write("modulusValues: {0}\n", JsonConvert.SerializeObject(modulusValues, Formatting.Indented));

            // Record the fractal value collection for this ray 
            TracedRay tracedRay = new TracedRay(externalPoints, modulusValues, angleValues, distanceValues);

            // Add this ray to the ray map in the sphere
            sphere.RayMap[rayCountX, rayCountY] = tracedRay.RayData;
        }
    }

    sealed public class ImageRendering
    {
        #region Member Variables

        private clsSphere sphere;
        private ImageDisplay imageDisplay;
        private int[] raysProcessed;
        private int[] linesProcessed;

        //  Sequence playback
        // private static int m_CurrentKey;		// Current key frame for sequence
        // private static int m_FrameCount;		// Frame to display for current key

        #endregion

        #region Delegates

        public delegate void UpdateRowStatusDelegate(int[] rowArray, int totalLines);
        public event UpdateRowStatusDelegate updateRowStatus;

        public delegate void UpdateRayStatusDelegate(int[] rayArray, int totalRays);
        public event UpdateRayStatusDelegate updateRayStatus;

        #endregion

        #region Class Properties

        // public static Size SeqSize
        // {
        //     get { return m_SeqSize; }
        //     set { m_SeqSize = value; }
        // }

        //public static float ScaleValue
        //{
        //    get { return m_ScaleValue; }
        //    set { m_ScaleValue = value; }
        //}

        // public static int CurrentKey
        // {
        //     get { return m_CurrentKey; }
        //     set { m_CurrentKey = value; }
        // }

        // public static int FrameCount
        // {
        //     get { return m_FrameCount; }
        //     set { m_FrameCount = value; }
        // }

        #endregion

        #region DLL Imports

        [DllImport("Unmanaged.dll")]
        static extern void InitSphere(float fBailout, float Resolution,
                                      float Latitude, float Longitude,
                                      float Radius, float verticalView, float horizontalView, float[,] PositionMatrix);

        #endregion

        public ImageRendering()
        {
        }

        public void InitialiseSphere()
        {
            // Set the ImageRendering sphere to the global sphere
            sphere = Model.Globals.Sphere;

            // Initialise the sphere in the dll from the ImageRendering sphere
            InitSphere(sphere.settings.Bailout, sphere.settings.AngularResolution,
                       sphere.settings.CentreLatitude, sphere.settings.CentreLongitude,
                       sphere.settings.Radius, sphere.settings.VerticalView, sphere.settings.HorizontalView, sphere.settings.PositionMatrix);

            RayTracingParams rayParams;

            rayParams.startDistance = sphere.settings.Radius;
            rayParams.increment = sphere.settings.SamplingInterval[sphere.settings.ActiveIndex];
            rayParams.smoothness = sphere.settings.SurfaceSmoothing;
            rayParams.surfaceThickness = sphere.settings.SurfaceThickness;
            rayParams.bailout = sphere.settings.Bailout;
            rayParams.rayPoints = (int)(sphere.settings.MaxSamples[sphere.settings.ActiveIndex] * sphere.settings.SamplingInterval[sphere.settings.ActiveIndex]);
            rayParams.maxSamples = sphere.settings.MaxSamples[sphere.settings.ActiveIndex];
            rayParams.boundaryInterval = sphere.settings.BoundaryInterval;
            rayParams.binarySearchSteps = sphere.settings.BinarySearchSteps[sphere.settings.ActiveIndex];
            rayParams.activeIndex = sphere.settings.ActiveIndex;

            bool success = InitializeGPU(ref rayParams);
            if (!success)
            {
                Console.WriteLine("Failed to initialize GPU parameters");
            }

            success = CopyMatrix(sphere.settings.PositionMatrix);
            if (!success)
            {
                Console.WriteLine("Failed to set transform matrix");
            }

            // Verify the matrix
            try
            {
                float[,] verifiedMatrix = VerifyMatrix();

                Console.WriteLine("Verify matrix");
                // TODO: Compare verifiedMatrix with positionMatrix to ensure they match
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error verifying matrix: {ex.Message}");
            }
        }

        public static bool CopyMatrix(float[,] positionMatrix)
        {
            int rows = positionMatrix.GetLength(0);
            int cols = positionMatrix.GetLength(1);
            float[] flatMatrix = new float[rows * cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    flatMatrix[i * cols + j] = positionMatrix[i, j];
                }
            }

            return CopyTransformationMatrix(flatMatrix);
        }

        public static float[,] VerifyMatrix()
        {
            int DimTotal = 5;

            float[] flatOutput = new float[DimTotal * 6];
            bool success = VerifyTransformationMatrix(flatOutput);

            if (!success)
            {
                throw new Exception("Failed to verify transform matrix");
            }

            float[,] output = new float[DimTotal, 6];
            for (int i = 0; i < DimTotal; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    output[i, j] = flatOutput[i * 6 + j];
                }
            }

            return output;
        }

        public async Task<bool> PerformRayTracing()
        {
            return await Task.Run(() =>
            {
                imageDisplay = new ImageDisplay();
                imageDisplay.LockBitmap();

                try
                {
                    // Initialise ray map
                    sphere.InitialiseRayMap();

                    int totalLines = (int)(sphere.settings.VerticalView / sphere.settings.AngularResolution);
                    int totalRays = (int)(sphere.settings.HorizontalView / sphere.settings.AngularResolution);

                    raysProcessed = new int[totalRays];
                    linesProcessed = new int[totalLines];

                    PerformParallel(totalLines, totalRays);
                }
                catch (Exception e)
                {
                    MessageBox.Show("Error: " + e.Message, "Raytracing Error");
                }

                imageDisplay.UnlockBitmap();
                // Store the image in the sphere
                sphere.ViewportImage = imageDisplay.GetBitmap();
                return true;
            });
        }

        private void PerformParallel(int totalLines, int totalRays)
        {
            RayProcessing[,] rayProc = new RayProcessing[totalRays, totalLines];

            for (int countY = 0; countY < totalLines; countY++)
            {
                for (int countX = 0; countX < totalRays; countX++)
                {
                    // Instantiate a class for this thread
                    rayProc[countX, countY] = new RayProcessing();
                }
            }
            // For each latitude line in the viewport
            Parallel.For(0, totalLines, rayCountY =>
            {
                // For each longitude point on this line
                Parallel.For(0, totalRays, rayCountX =>
                {
                    try
                    {
                        // Perform raytracing
                        rayProc[rayCountX, rayCountY].ProcessRay(sphere, rayCountX, rayCountY);

                        //RayCompleted(rayCountY * totalRays + rayCountX);
                    }
                    catch
                    { }
                });
                RowCompleted(rayCountY);
            });

            Parallel.For(0, totalLines, rayCountY =>
            {
                // For each longitude point on this line
                Parallel.For(0, totalRays, rayCountX =>
                 {
                     try
                     {
                         // Set the colour and display the point
                         TracedRay tracedRay = SetRayColour(sphere, rayCountX, rayCountY);
                         imageDisplay.updateImage(rayCountX, rayCountY, tracedRay.bmiColors);
                     }
                     catch
                     { }
                 });
            });
        }

        public async Task<bool> Redisplay()
        {
            return await Task.Run(() =>
            {
                //picImage.Image = new Bitmap(picImage.Image.Width, picImage.Image.Height);
                if (Model.Globals.Sphere.RayMap != null)
                {
                    int totalLines = (int)(sphere.settings.VerticalView / sphere.settings.AngularResolution);
                    linesProcessed = new int[totalLines];

                    try
                    {
                        // for (int lineIndex = 0; lineIndex < totalLines; lineIndex++)
                        Parallel.For(0, totalLines, lineIndex =>
                        {
                            // Set pixel colours for this line
                            Redisplay(lineIndex);
                            RowCompleted((int)lineIndex);
                        });
                        sphere.ViewportImage = imageDisplay.GetBitmap();
                    }
                    catch (Exception e)
                    {
                        MessageBox.Show("Error: " + e.Message, "Redisplay Error");
                    }
                }
                return true;
            });
        }

        private void ProgressChanged(int rayCountX, int rayCountY, TracedRay ray)
        {
            if (ray != null)
            {
                int totalRays = (int)(sphere.settings.HorizontalView / sphere.settings.AngularResolution);

                // If current row is still being processed
                if (rayCountX < totalRays - 1)
                {
                    imageDisplay.updateImage(rayCountX, rayCountY, ray.bmiColors);
                }
            }
        }

        private void RayCompleted(int rayIndex)
        {
            raysProcessed[rayIndex] = 1;
            int totalRays = (int)(sphere.settings.HorizontalView / sphere.settings.AngularResolution);

            // Call the UpdateStatus function in Main
            updateRayStatus(raysProcessed, totalRays);
        }

        private void RowCompleted(int lineIndex)
        {
            linesProcessed[lineIndex] = 1;
            int totalLines = (int)(sphere.settings.VerticalView / sphere.settings.AngularResolution);

            // Call the UpdateStatus function in Main
            updateRowStatus(linesProcessed, totalLines);
        }

        public void Redisplay(int rayCountY)
        {
            int totalRays = (int)(sphere.settings.HorizontalView / sphere.settings.AngularResolution);

            // For each longitude point on this line
            for (int rayCountX = 0; rayCountX < totalRays; rayCountX++)
            {
                // Display the point on this line of latitude
                TracedRay tracedRay = SetRayColour(sphere, rayCountX, rayCountY);
                ProgressChanged(rayCountX, rayCountY, tracedRay);
            }
        }

        private TracedRay SetRayColour(clsSphere sphere, int rayCountX, int rayCountY)
        {
            if (rayCountX >= sphere.RayMap.GetUpperBound(0) || rayCountY >= sphere.RayMap.GetUpperBound(1))
            {
                return null;
            }
            // Get the ray from the ray map
            TracedRay.RayDataType rayData = sphere.RayMap[rayCountX++, rayCountY];
            TracedRay tracedRay = new TracedRay(rayData.ExternalPoints, rayData.ModulusValues, rayData.AngleValues, rayData.DistanceValues);

            // Calculate the tilt values from the previous rays
            if (rayCountX > 0)
            {
                tracedRay.xTiltValues = sphere.addTiltValues(tracedRay, rayCountX - 1, rayCountY);
            }
            if (rayCountY > 0)
            {
                tracedRay.yTiltValues = sphere.addTiltValues(tracedRay, rayCountX, rayCountY - 1);
            }

            if (tracedRay != null)
            {
                // Convert the fractal value collection into an rgb colour value
                tracedRay.SetColour();
            }
            return tracedRay;
        }

        public Bitmap GetBitmap()
        {
            return imageDisplay.GetBitmap();
        }

        //private void extendBoundaries(int row)
        //{
        //    // Start with the first ray
        //    int rayNumber = 0;

        //    // For each longitude point on this line working forwards
        //    for (float longitude = sphere.settings.LongitudeStart - sphere.settings.Resolution;
        //        longitude > sphere.settings.LongitudeEnd; longitude -= sphere.settings.Resolution)
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
        //    for (float longitude = sphere.settings.LongitudeEnd + sphere.settings.Resolution;
        //        longitude < sphere.settings.LongitudeStart; longitude += sphere.settings.Resolution)
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
        //    List<float> newBoundaries = targetRay.Boundaries;
        //    int sourcePos = -1;
        //    int targetPos = 0;

        //    // For each source boundary position value
        //    foreach (float sourcePosition in sourceRay.Boundaries)
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

        //        if (Single.IsPositiveInfinity(targetRay.Boundary(targetPos)))
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
        //        if (Math.Abs(sourceRay.Angle(sourcePos) - targetRay.Angle(targetPos)) < sphere.settings.BoundaryInterval)
        //        {
        //            // Go on to the next source boundary position
        //            continue;
        //        }

        //        // Set increment to the distance between the target and source points
        //        float separation = sourcePosition - targetRay.Boundary(targetPos);

        //        float Modulus, Angle;
        //        bool externalPoint;

        //        // Perform binary search between the target and the source points, to determine new boundary position
        //        float newPosition = FindBoundary(separation, sphere.settings.BinarySearchSteps, sourcePosition, targetRay.Angle(targetPos),
        //                              sphere.settings.BoundaryInterval, &externalPoint, &Modulus, &Angle);

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
