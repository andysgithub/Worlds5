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
        private int nextLineToProcess = 0;
        BackgroundWorker[] lineThread = new BackgroundWorker[Globals.TOTAL_THREADS];
        private bool redisplayPending = false;

        public delegate void RayDataDelegate(double latitude, double longitude, Model.Globals.RGBQUAD rayColors);
        public delegate void UpdateBitmapDelegate(int rowCount);
        public delegate void FrameCompletedDelegate();
        public event RayDataDelegate returnRayData;
        public event UpdateBitmapDelegate updateBitmap;
        public event FrameCompletedDelegate frameCompleted;

        public NodeController()
        {
            // Initialise all threads for line processing
            InitialiseThreads();
        }

        private void InitialiseThreads()
        {
            for (int i = 0; i < Globals.TOTAL_THREADS; i++)
            {
                lineThread[i] = new BackgroundWorker();
                lineThread[i].DoWork += new DoWorkEventHandler(bwThread_DoWork);
                lineThread[i].ProgressChanged += new ProgressChangedEventHandler(bwThread_ProgressChanged);
                lineThread[i].RunWorkerCompleted += new RunWorkerCompletedEventHandler(bwThread_RunWorkerCompleted);
                lineThread[i].WorkerReportsProgress = true;
                lineThread[i].WorkerSupportsCancellation = true;
            }
        }

        public void PerformRayTracing()
        {
            try
            {
                // Initialise ray map
                Model.Globals.Sphere.InitialiseRayMap();
                clsSphere sphere = Model.Globals.Sphere;
                double totalLines = (int)(sphere.VerticalView / sphere.AngularResolution);

                for (int lineIndex = 0; lineIndex < totalLines; lineIndex++) {
                    try
                    {
                        // Perform raytracing
                        ProcessLine(lineIndex, DisplayOption.Start);
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

            for (int lineIndex = 0; lineIndex < totalLines; lineIndex++)
            {
                // Perform raytracing using the background worker thread
                if (Model.Globals.Sphere.RayMap == null)
                {
                    ProcessLine(lineIndex, DisplayOption.Start);
                }
                else
                {
                    ProcessLine(lineIndex, DisplayOption.Redisplay);
                }
            }
        }

        private void ProcessLine(int lineIndex, DisplayOption option)
        {
            RenderThread rt = new RenderThread(ImageRendering.Bailout);

            if (option == DisplayOption.Start)
            {
                rt.RenderRays(lineIndex);
            }
            else
            {
                rt.Redisplay(lineIndex);
            }
        }

        // private void bwThread_ProgressChanged(object sender, ProgressChangedEventArgs e)
        // {
        //     object[] Args = (object[])e.UserState;
        //     int rowCount = e.ProgressPercentage;

        //     if (rowCount == -1)
        //     {
        //         // Current ray has been processed

        //         double latitude = (double)Args[0];
        //         double longitude = (double)Args[1];
        //         TracedRay ray = (TracedRay)Args[2];

        //         // Send ray colours to front end
        //         if (returnRayData != null)
        //             returnRayData(latitude, longitude, ray.bmiColors);
        //     }
        //     else
        //     {
        //         // All rays have has been completed for this row
        //         if (updateBitmap != null)
        //             updateBitmap(rowCount);
        //     }
        // }

        private void RowCompleted(int lineIndex, DisplayOption displayOption)
        {
            if (displayOption == DisplayOption.Cancel)
            {
                //staStatus.Items[0].Text = "Ray tracing cancelled.";

                if (redisplayPending)
                {
                    redisplayPending = false;
                    // Perform raytracing using the background worker thread
                    lineThread[threadIndex].RunWorkerAsync(new object[] { threadIndex, DisplayOption.Redisplay });
                    return;
                }
            }

            //clsSphere sphere = Model.Globals.Sphere;
            //double totalLines = (int)(sphere.VerticalView / sphere.AngularResolution);

            // If a line is still available to process
            // if (nextLineToProcess < totalLines)
            // {
            //     // Perform raytracing
            //     ProcessLine(nextLineToProcess, displayOption);
            // }
            // else
            // {

                // Frame has completed processing
                if (frameCompleted != null)
                    frameCompleted();
            // }
        }
    }
}
