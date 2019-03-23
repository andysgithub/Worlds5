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

        // Matric containing a single transformation
        private double[,] manip = new double[6, 6];
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

        public void PerformRotation(double[] centreCoords, double[,] angles, int totalFrames, string basePath)
        {
            // TODO: This should be in its own thread

            double[] Position = new double[DimTotal];
            for (int col = 0; col < DimTotal; ++col)
            {
                Position[col] = sphere.PositionMatrix[DimTotal, col];
            }

            // For each frame in the sequence
            for (int frameCount = 0; frameCount < totalFrames; frameCount++)
            {
                // Translate to the centre coords
                for (int col = 0; col < DimTotal; ++col)
                {
                    Position[col] = -Position[col];
                }
                SetTranslation(2, Position);
                PreMulT();

                for (int axis1 = 1; axis1 < 5; axis1++)
                {
                    for (int axis2 = 2; axis2 < 6; axis2++)
                    {
                        // If rotation is set for this plane
                        if (angles[axis1, axis2] > 0)
                        {
                            // Rotate by the given angle
                            SetRotation(axis1-1, axis2-1, angles[axis1, axis2]);
                            PreMulR();
                        }
                    }
                }

                // Translate back from the centre
                for (int col = 0; col < DimTotal; ++col)
                {
                    Position[col] = -Position[col];
                }
                SetTranslation(2, Position);
                PreMulT();

                PathName = String.Format("{0}_{1:000}.{2}", basePath, frameCount, extension);
                if (!File.Exists(PathName))
                {
                    imageRendering.PerformRayTracing();
                    SaveFrame();
                    Application.DoEvents();
                }
                else
                {
                    picImage.Image = new Bitmap(PathName);
                    Application.DoEvents();
                }
            }
        }

        /// <summary>
        /// Callback to save a frame after processing.
        /// </summary>
        /// <param name="rowCount"></param>
        private void SaveFrame()
        {
            // File this image
            picImage.Image.Save(PathName, format);
        }

        /// <summary>
        /// Initialise the manipulation matrix.
        /// </summary>
        void ManipInit()
        {
            int row, col;

            for (row = 0; row <= 5; ++row)
            {
                for (col = 0; col <= 5; ++col)
                {
                    manip[row, col] = 0;
                    manip[row, col] = 0;
                }
                manip[row, row] = 1;
            }
        }

        private void SetRotation(int Axis1, int Axis2, double Angle)
        {
            ManipInit();

            int a, b;

            if (Axis1 < Axis2)
            {
                a = Axis1;
                b = Axis2;
            }
            else
            {
                a = Axis2;
                b = Axis1;
            }

            manip[a, a] = Math.Cos(Angle);
            manip[b, b] = Math.Cos(Angle);
            manip[b, a] = Math.Sin(Angle);
            manip[a, b] = -Math.Sin(Angle);

            if (((b - a) == 2) || ((b - a) == 4))
            {
                manip[b, a] = -manip[b, a];
                manip[a, b] = -manip[a, b];
            }
        }

        private void SetTranslation(int Axis, double[] Position)
        {
            ManipInit();
            for (int col = 0; col < DimTotal; ++col)
            {
                manip[DimTotal, col] = Position[col];
            }
        }

        // Matrix pre-multiply for rotations
        private void PreMulR()
        {
            double[,] temp = new double[6, 6];

            for (int row = 0; row < DimTotal; row++)
            {
                for (int col = 0; col < DimTotal; ++col)
                {
                    temp[row, col] = 0;

                    for (int count = 0; count < DimTotal; count++)
                    {
                        temp[row, col] += manip[row, count] * sphere.PositionMatrix[count, col];
                    }
                }
            }

            for (int row = 0; row < DimTotal; row++)
            {
                for (int col = 0; col < DimTotal; col++)
                {
                    sphere.PositionMatrix[row, col] = temp[row, col];
                }
            }
        }

        // Matrix pre-multiply for translations
        void PreMulT()
        {
            for (int col = 0; col < DimTotal; col++)
            {
                for (int row = 0; row < DimTotal; row++)
                {
                    sphere.PositionMatrix[DimTotal, col] += manip[DimTotal, row] * sphere.PositionMatrix[row, col];
                }
            }
        }

        // Matrix post-multiply for translations
        void PostMulT()
        {
            for (int col = 0; col < DimTotal; col++)
            {
                for (int row = 0; row < DimTotal; row++)
                {
                    sphere.PositionMatrix[DimTotal, col] += sphere.PositionMatrix[DimTotal, row] * manip[row, col];
                }
            }
        }

        //	Matrix post-multiply for rotations
        void PostMulR()
        {
            double[,] temp = new double[6, 6];

            for (int row = 0; row < DimTotal; row++)
            {
                for (int col = 0; col < DimTotal; col++)
                {
                    temp[row, col] = 0;

                    for (int count = 0; count < DimTotal; ++count)
                    {
                        temp[row, col] += sphere.PositionMatrix[row, count] * manip[count, col];
                    }
                }
            }
            for (int row = 0; row < DimTotal; row++)
            {
                for (int col = 0; col < DimTotal; col++)
                {
                    sphere.PositionMatrix[row, col] = temp[row, col];
                }
            }
        }
    }
}
