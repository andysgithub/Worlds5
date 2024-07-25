using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Forms;
using Model;
using Syncfusion.Windows.Forms.Tools;
using static Model.TracedRay;
using static Worlds5.RayProcessing;

namespace Worlds5
{
<<<<<<< HEAD
=======
    sealed public class RayProcessing
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct RayTracingParams
        {
            public double startDistance;
            public double increment;
            public double smoothness;
            public double surfaceThickness;
            public float bailout;
            public int rayPoints;
            public int maxSamples;
            public double boundaryInterval;
            public int binarySearchSteps;
            public int activeIndex;
        }

        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern int TraceRay(double startDistance, double increment, double smoothness, double surfaceThickness,
            double xFactor, double yFactor, double zFactor, float bailout,
            int[] externalsArray, float[] valuesArray, float[] anglesArray, double[] distancesArray,
            int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
            int activeIndex);

        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern bool InitializeGPU(ref RayTracingParams parameters);

        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern bool CopyTransformationMatrix([In] double[] positionMatrix);

        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern bool VerifyTransformationMatrix([Out] double[] output);

        //[DllImport("Unmanaged.dll")]
        //static extern double[] ImageToFractalSpace (double startDistance, double xFactor, double yFactor, double zFactor);

        private int[] externalPoints;
        private float[] modulusValues;
        private float[] angleValues;
        private double[] distanceValues;

        public RayProcessing()
        {
            externalPoints = new int[100];
            modulusValues = new float[100];
            angleValues = new float[100];
            distanceValues = new double[100];
        }

        // Trace the ray on this latitude line
        public void ProcessRay(clsSphere sphere, int rayCountX, int rayCountY)
        {
            clsSphere.Settings settings = sphere.settings;
            double latitude = settings.LatitudeStart - rayCountY * settings.AngularResolution;
            double longitude = settings.LongitudeStart - rayCountX * settings.AngularResolution;
            int i = settings.ActiveIndex;
            int rayPoints = (int)(settings.MaxSamples[i] * settings.SamplingInterval[i]);

            double latRadians = latitude * Globals.DEG_TO_RAD;
            double longRadians = longitude * Globals.DEG_TO_RAD;

            double xFactor = Math.Cos(latRadians) * Math.Sin(-longRadians);
            double yFactor = Math.Sin(latRadians);
            double zFactor = Math.Cos(latRadians) * Math.Cos(-longRadians);

            // Set the start distance to the sphere radius
            double startDistance = settings.Radius;

            // If clipping is enabled
            if (settings.UseClipping)
            {
                // Get the 5D coordinates for the intersection between this vector and the clipping plane
                double distance = Clipping.CalculateDistance(latRadians, longRadians, settings.ClippingAxes, settings.ClippingOffset);

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

            // Record the fractal value collection for this ray 
            TracedRay tracedRay = new TracedRay(externalPoints, modulusValues, angleValues, distanceValues);

            // Add this ray to the ray map in the sphere
            sphere.RayMap[rayCountX, rayCountY] = tracedRay.RayData;
        }
    }

>>>>>>> 37878f5799e21f6fae4ca20707d2a6b276555608
    sealed public class ImageRendering
    {
        #region Member Variables

        private clsSphere sphere;
        private ImageDisplay imageDisplay;

        //  Sequence playback
        // private static int m_CurrentKey;        // Current key frame for sequence
        // private static int m_FrameCount;        // Frame to display for current key

        #endregion

        #region Delegates

        public delegate void UpdateRayStatusDelegate(int count, int totalRays);
        public event UpdateRayStatusDelegate updateRayStatus;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate void ProgressCallback(int rayCount, int rowCount, IntPtr rayDataPtr);

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

        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern void InitSphere(float[,] PositionMatrix);

        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ProcessRays(RayTracingParams rayParams, RenderingParams renderParams, int raysPerLine, int totalLines, ProgressCallback callback);

        #endregion

        public ImageRendering()
        {
        }

        public void InitialiseSphere()
        {
            // Set the ImageRendering sphere to the global sphere
            sphere = Model.Globals.Sphere;

            // Initialise the sphere in the dll from the ImageRendering sphere
            InitSphere(sphere.settings.PositionMatrix);

<<<<<<< HEAD
            RayTracingParams rayParams = new RayTracingParams(sphere.settings);
            RenderingParams renderParams = new RenderingParams(sphere.settings);
=======
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
                double[,] verifiedMatrix = VerifyMatrix();
                // TODO: Compare verifiedMatrix with positionMatrix to ensure they match
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error verifying matrix: {ex.Message}");
            }
>>>>>>> 37878f5799e21f6fae4ca20707d2a6b276555608
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
                    int raysPerLine = (int)(sphere.settings.HorizontalView / sphere.settings.AngularResolution);

                    PerformParallel(totalLines, raysPerLine);
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

        private void PerformParallel(int totalLines, int raysPerLine)
        {
            RayTracingParams rayParams = new RayTracingParams(sphere.settings);
            RenderingParams renderParams = new RenderingParams(sphere.settings);

            int rayCount = 0;

            ProgressCallback progressCallback = (rayCountX, rayCountY, rayDataPtr) =>
            {
                RayDataTypeIntermediate intermediateData = Marshal.PtrToStructure<RayDataTypeIntermediate>(rayDataPtr);

                RayDataType rayData = Helpers.ConvertFromIntermediate(intermediateData);
                sphere.RayMap[rayCountX, rayCountY] = rayData;

                if (rayCount++ % 100 == 0) RayCompleted(rayCount);
            };

            ProcessRays(rayParams, renderParams, raysPerLine, totalLines, progressCallback);
            RayCompleted(raysPerLine * totalLines);

            Parallel.For(0, totalLines, rayCountY =>
            {
                // For each longitude point on this line
                Parallel.For(0, raysPerLine, rayCountX =>
                 {
                     try
                     {
                         // Set the colour and display the point
                         TracedRay tracedRay = SetRayColour(sphere, renderParams, rayCountX, rayCountY);
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
                    int rayCount = 0;

                    try
                    {
                        Parallel.For(0, totalLines, lineIndex =>
                        {
                            // Set pixel colours for this line
                            Redisplay(lineIndex);
                            if (rayCount++ % 100 == 0) RayCompleted(rayCount);
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
                int raysPerLine = (int)(sphere.settings.HorizontalView / sphere.settings.AngularResolution);

                // If current row is still being processed
                if (rayCountX < raysPerLine - 1)
                {
                    imageDisplay.updateImage(rayCountX, rayCountY, ray.bmiColors);
                }
            }
        }

        private void RayCompleted(int rayIndex)
        {
            int raysPerLine = (int)(sphere.settings.HorizontalView / sphere.settings.AngularResolution);
            int totalLines = (int)(sphere.settings.VerticalView / sphere.settings.AngularResolution);

            // Call the UpdateStatus function in Main
            updateRayStatus(rayIndex, raysPerLine * totalLines);
        }

        public void Redisplay(int rayCountY)
        {
            int raysPerLine = (int)(sphere.settings.HorizontalView / sphere.settings.AngularResolution);
            RenderingParams renderParams = new RenderingParams(sphere.settings);

            // For each longitude point on this line
            for (int rayCountX = 0; rayCountX < raysPerLine; rayCountX++)
            {
                // Display the point on this line of latitude
                TracedRay tracedRay = SetRayColour(sphere, renderParams, rayCountX, rayCountY);
                ProgressChanged(rayCountX, rayCountY, tracedRay);
            }
        }

        private TracedRay SetRayColour(clsSphere sphere, RenderingParams renderParams, int rayCountX, int rayCountY)
        {
            if (rayCountX >= sphere.RayMap.GetUpperBound(0) || rayCountY >= sphere.RayMap.GetUpperBound(1))
            {
                return null;
            }
            // Get the ray from the ray map
            TracedRay.RayDataType rayData = sphere.RayMap[rayCountX++, rayCountY];
            TracedRay tracedRay = new TracedRay(rayData.ExternalPoints, rayData.ModulusValues, rayData.AngleValues, rayData.DistanceValues, renderParams);

            // Calculate the tilt values from the previous rays
            if (rayCountX > 0)
            {
                tracedRay.xTiltValues = sphere.addTiltValues(tracedRay, renderParams, rayCountX - 1, rayCountY);
            }
            if (rayCountY > 0)
            {
                tracedRay.yTiltValues = sphere.addTiltValues(tracedRay, renderParams, rayCountX, rayCountY - 1);
            }

            if (tracedRay != null && rayData.ModulusValues != null)
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
    }
}
