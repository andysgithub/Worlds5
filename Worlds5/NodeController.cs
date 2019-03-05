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
                nextLineToProcess = 0;

                for (int threadIndex = 0; threadIndex < Globals.TOTAL_THREADS; threadIndex++)
                {
                    try
                    {
                        // Perform raytracing using the background worker thread
                        lineThread[threadIndex].RunWorkerAsync(new object[] { threadIndex, DisplayOption.Start });
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
            nextLineToProcess = 0;
            //picImage.Image = new Bitmap(picImage.Image.Width, picImage.Image.Height);

            for (int threadIndex = 0; threadIndex < Globals.TOTAL_THREADS; threadIndex++)
            {
                if (lineThread[threadIndex].IsBusy)
                {
                    // Cancel any current display process
                    lineThread[threadIndex].CancelAsync();
                    redisplayPending = true;
                }
                else
                {
                    // Perform raytracing using the background worker thread
                    if (Model.Globals.Sphere.RayMap == null)
                    {
                        lineThread[threadIndex].RunWorkerAsync(new object[] { threadIndex, DisplayOption.Start });
                    }
                    else
                    {
                        lineThread[threadIndex].RunWorkerAsync(new object[] { threadIndex, DisplayOption.Redisplay });
                    }
                }
            }
        }

        private void bwThread_DoWork(object sender, DoWorkEventArgs e)
        {
            object[] args = (object[])e.Argument;
            int threadIndex = (int)args[0];

            // Get next available line to process
            int lineIndex = nextLineToProcess++;

            DisplayOption option = DisplayOption.Start;
            if (args[1] != null)
            {
                option = (DisplayOption)args[1];
            }

            RenderThread rt = new RenderThread(ImageRendering.Bailout, lineThread[threadIndex]);

            if (option == DisplayOption.Start)
            {
                rt.RenderRays(lineIndex);
            }
            else
            {
                rt.Redisplay(lineIndex);
            }

            if (lineThread[threadIndex].CancellationPending)
            {
                option = DisplayOption.Cancel;
            }

            e.Result = new object[] { threadIndex, option };
        }

        private void bwThread_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            object[] Args = (object[])e.UserState;
            int rowCount = e.ProgressPercentage;

            if (rowCount == -1)
            {
                // Current ray has been processed

                double latitude = (double)Args[0];
                double longitude = (double)Args[1];
                TracedRay ray = (TracedRay)Args[2];

                // Send ray colours to front end
                if (returnRayData != null)
                    returnRayData(latitude, longitude, ray.bmiColors);
            }
            else
            {
                // All rays have has been completed for this row
                if (updateBitmap != null)
                    updateBitmap(rowCount);
            }
        }

        private void bwThread_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            object[] Args = (object[])e.Result;
            int threadIndex = (int)Args[0];
            DisplayOption displayOption = (DisplayOption)Args[1];

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

            clsSphere sphere = Model.Globals.Sphere;
            double totalLines = (int)(sphere.VerticalView / sphere.AngularResolution);

            // If a line is still available to process
            if (nextLineToProcess < totalLines)
            {
                // Perform raytracing using the background worker thread
                lineThread[threadIndex].RunWorkerAsync(new object[] { threadIndex, displayOption });
            }
            else
            {
                lineThread[threadIndex].CancelAsync();

                for (int i = 0; i < Globals.TOTAL_THREADS; i++)
                {
                    if (!lineThread[i].CancellationPending)
                    {
                        return;
                    }
                }

                // Frame has completed processing
                if (frameCompleted != null)
                    frameCompleted();
            }
        }
    }
}
