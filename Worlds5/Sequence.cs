using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Windows.Forms;
using Model;

namespace Worlds5
{
    class Sequence
    {
        // The thread to use for generating each frame
        //private BackgroundWorker bwImageThread;
        private PictureBox picImage;
        private ImageFormat format;
        private string extension;
        private ImageRendering imageRendering;
        private string PathName;

        private clsSphere sphere = Model.Globals.Sphere;
        private int DimTotal = 5;	// Total number of dimensions used

        public Sequence(ImageRendering imageRendering, PictureBox picImage, ImageFormat format)
        {
            this.imageRendering = imageRendering;

            this.picImage = picImage;
            this.format = format;

            if (format == ImageFormat.Jpeg)
                this.extension = "jpg";
            else if (format == ImageFormat.Tiff)
                this.extension = "tif";
            else if (format == ImageFormat.Bmp)
                this.extension = "bmp";
            else if (format == ImageFormat.Png)
                this.extension = "png";
        }

        public async void PerformRotation(double[] centreCoords, double[,] angles, double[] sphereRadius, int totalFrames, string basePath)
        {
            // TODO: This should be in its own thread

            double[] Position = new double[DimTotal];
            for (int col = 0; col < DimTotal; ++col)
            {
                Position[col] = sphere.settings.PositionMatrix[DimTotal, col];
            }

            // For each frame in the sequence
            for (int frameCount = 0; frameCount < totalFrames; frameCount++)
            {
                sphere.settings.Radius = sphereRadius[0] - (double)frameCount / (double)totalFrames * (sphereRadius[0] - sphereRadius[1]);

                // Translate to the centre coords
                for (int col = 0; col < DimTotal; ++col)
                {
                    Position[col] = -Position[col];
                }
                Transformation.SetTranslation(2, Position);
                Transformation.PreMulT();

                for (int axis1 = 1; axis1 < 5; axis1++)
                {
                    for (int axis2 = 2; axis2 < 6; axis2++)
                    {
                        // If rotation is set for this plane
                        if (angles[axis1, axis2] > 0)
                        {
                            // Rotate by the given angle
                            Transformation.SetRotation(axis1-1, axis2-1, angles[axis1, axis2]);
                            Transformation.PreMulR();
                        }
                    }
                }

                // Translate back from the centre
                for (int col = 0; col < DimTotal; ++col)
                {
                    Position[col] = -Position[col];
                }
                Transformation.SetTranslation(2, Position);
                Transformation.PreMulT();

                imageRendering.InitialiseSphere();

                PathName = String.Format("{0}_{1:000}.{2}", basePath, frameCount, extension);
                if (!File.Exists(PathName))
                {
                    await imageRendering.PerformRayTracing();
                    SaveFrame();
                }
                else
                {
                    picImage.Image = new Bitmap(PathName);
                }
            }
        }

        /// <summary>
        /// Callback to save a frame after processing.
        /// </summary>
        /// <param name="rowCount"></param>
        private void SaveFrame()
        {
            picImage.Image = imageRendering.GetBitmap();
            // File this image
            picImage.Image.Save(PathName, format);
        }
    }
}
