using System;
using System.ComponentModel;
using System.Drawing;
using System.Windows.Forms;
using System.Threading;
using System.Collections;
using DataClasses;
using System.Runtime.InteropServices;

namespace PatchServer
{
    /// <summary>
    /// Summary description for Fractal.
    /// </summary>
    public class RenderPatch
    {
        [DllImport("Unmanaged.dll")]
        static extern void GenerateImage(Globals.vertex v0, Globals.vertex v1, Globals.vertex v2, int m_PatchNo, ref Globals.PixelData PixelData);

        public RenderPatch(DataClasses.Globals.vertex[] vertices, int PatchNo, BackgroundWorker bwThread)
        {
            Globals.PixelData[] PixelData = new Globals.PixelData[1000]; 

            GenerateImage(vertices[0], vertices[1], vertices[2], PatchNo, ref PixelData[0]);
            // Report progress for the resulting bitmap 
            bwThread.ReportProgress(100, new object[] { PatchNo, PixelData });
        }
    }
}
