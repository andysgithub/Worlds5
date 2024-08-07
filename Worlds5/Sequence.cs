﻿using System;
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

        public async void PerformRotation(float[] centreCoords, float[,] angles, float[] sphereRadius, int totalFrames, string basePath)
        {
            // For each frame in the sequence
            for (int frameCount = 0; frameCount < totalFrames; frameCount++)
            {
                sphere.settings.SphereRadius = sphereRadius[0] - (float)frameCount / (float)totalFrames * (sphereRadius[0] - sphereRadius[1]);

                Transformation.RotateImage(RotationCentre.Origin, angles);
                sphere.settings.PositionMatrix = Transformation.GetPositionMatrix();

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
