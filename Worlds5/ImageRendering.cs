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

            RayTracingParams rayParams = new RayTracingParams(sphere.settings);
            RenderingParams renderParams = new RenderingParams(sphere.settings);
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
