using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace DataClasses
{
    public class Patch
    {
        /// <summary>
        /// The patch class defines the triangular patches composing the sphere surface. 
        /// Each patch consists or a set of three vertices and a centroid coordinate, giving the position of the patch on the sphere.
        /// The patch also stores the pixel data of the image in that region.
        /// The reference vertices define an identical triangle with the centroid at coord 0,0,1. These reference vertices are used when
        /// generating and displaying the patch image, to identify the endpoints of the scan lines for each line of pixel data.
        /// The endpoints are then transformed to the correct sphere position before generating or displaying the pixels.
        /// </summary>
        public enum PatchStatus
        {
            Empty = 0,
            ReadyToDisplay,
            Displayed
        }

        public Globals.vertex[] Vertices = new Globals.vertex[3];
        private Globals.vertex m_Centroid;
        private Globals.PixelData[] m_PixelData;
        private PatchStatus m_PatchStatus = PatchStatus.Empty;

        public Patch(Globals.vertex Vertex1, Globals.vertex Vertex2, Globals.vertex Vertex3)
        {
            Vertices[0] = Vertex1;
            Vertices[1] = Vertex2;
            Vertices[2] = Vertex3;

            // Calculate and store the centroid coords from the vertices
            m_Centroid.X = (Vertex1.X + Vertex2.X + Vertex3.X) / 3;
            m_Centroid.Y = (Vertex1.Y + Vertex2.Y + Vertex3.Y) / 3;
            m_Centroid.Z = (Vertex1.Z + Vertex2.Z + Vertex3.Z) / 3;

            m_PixelData = new Globals.PixelData[1000];
        }

        public Globals.PixelData[] PixelData
        {
            get { return m_PixelData; }
            set { m_PixelData = value; }
        }

        public PatchStatus Status
        {
            get { return m_PatchStatus; }
            set { m_PatchStatus = value; }
        }

        public Globals.vertex Centroid
        {
            get { return m_Centroid; }
        }
    }
}
