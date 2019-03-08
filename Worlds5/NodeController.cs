using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Model;

namespace Worlds5
{
    /// <summary>
    /// This class controls each node on the network, and contains the line thread array for processing one line per thread.
    /// </summary>
    public class NodeController
    {
        private int linesProcessed = 0;
        private bool redisplayPending = false;

        public delegate void RayDataDelegate(double latitude, double longitude, Model.Globals.RGBQUAD rayColors);
        public delegate void UpdateBitmapDelegate(int rowCount);
        public delegate void FrameCompletedDelegate();
        public event RayDataDelegate returnRayData;
        public event UpdateBitmapDelegate updateBitmap;
        public event FrameCompletedDelegate frameCompleted;

        public NodeController()
        {
        }

        public void PerformRayTracing()
        {
            try
            {
                // Initialise ray map
                Model.Globals.Sphere.InitialiseRayMap();
                clsSphere sphere = Model.Globals.Sphere;
                ImageRendering rt = new ImageRendering();

                double totalLines = (int)(sphere.VerticalView / sphere.AngularResolution);

                for (int lineIndex = 0; lineIndex < totalLines; lineIndex++) {
                    try
                    {
                        // Perform raytracing
                        rt.RenderRays(lineIndex);
                    }
                    catch (InvalidOperationException ex)
                    { }
                }
            }
            catch (InvalidOperationException ex)
            { }
        }

        public void Redisplay()
        {
            //picImage.Image = new Bitmap(picImage.Image.Width, picImage.Image.Height);
            ImageRendering rt = new ImageRendering();
            
            for (int lineIndex = 0; lineIndex < totalLines; lineIndex++)
            {
                // Perform raytracing
                if (Model.Globals.Sphere.RayMap == null)
                {
                    rt.RenderRays(lineIndex);
                }
                else
                {
                    rt.Redisplay(lineIndex);
                }
            }
        }

        private void ProcessLine(int lineIndex, DisplayOption option)
        {
            ImageRendering rt = new ImageRendering();

            if (option == DisplayOption.Start)
            {
                rt.RenderRays(lineIndex);
            }
            else
            {
                rt.Redisplay(lineIndex);
            }
            RowCompleted(lineIndex, option);
        }

        private void ProgressChanged(int rayCountX, int rayCountY, TracedRay ray)
        {
            clsSphere sphere = Model.Globals.Sphere;
            int totalRays = (sphere.LongitudeStart - sphere.LongitudeEnd) / sphere.AngularResolution;

            // If current row is still being processed
            if (rayCountX < totalRays)
            {
                // Send ray colours to front end
                if (returnRayData != null) {
                    // Get lat/long from rayCountX/Y
                    double latitude = sphere.LatitudeStart - rayCountY * sphere.AngularResolution;
                    double longitude = sphere.LongitudeStart - rayCountX * sphere.AngularResolution;

                    // Call GetRayData in Main via the RayDataDelegate
                    returnRayData(latitude, longitude, ray.bmiColors);
                }
            }
            else
            {
                // All rays have has been completed for this row
                if (updateBitmap != null) {
                    // Call UpdateBitmap in Main via the UpdateBitmapDelegate
                    updateBitmap(rayCountY);
                }
            }
        }

        private void RowCompleted(int lineIndex, DisplayOption displayOption)
        {
            if (displayOption == DisplayOption.Cancel)
            {
                //staStatus.Items[0].Text = "Ray tracing cancelled.";

                if (redisplayPending)
                {
                    redisplayPending = false;
                    // Perform raytracing
                    ProcessLine(lineIndex, DisplayOption.Redisplay);
                    return;
                }
            }

            clsSphere sphere = Model.Globals.Sphere;
            double totalLines = (int)(sphere.VerticalView / sphere.AngularResolution);
            linesProcessed++;

            if (linesProcessed >= totalLines)
            {
                // Frame has completed processing
                if (frameCompleted != null)
                    // Call SaveFrame in Sequence via the FrameCompletedDelegate
                    frameCompleted();
            }
        }
    }
}
