using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Model;
using Accord.Video.FFMPEG;

namespace Worlds5
{
    public enum DisplayOption
    {
        None = 0,
        Start,
        Redisplay,
        Cancel
    }

    public partial class Main : Form
    {
        private bool bResizing = false;
        private DisplayStatus m_DisplayStatus = DisplayStatus.None;
        private string currentAddress = string.Empty;
        private ImageRendering imageRendering = null;

        [DllImport("Unmanaged.dll")]
        static extern void InitBitmap(int Width, int Height);
        [DllImport("Unmanaged.dll")]
        static extern void SetViewingAngle(double Latitude, double Longitude);

        private enum DisplayStatus
        {
            None = 0,
            InProgress,
            Completed,
            Stopped
        }

        public Main()
        {
            InitializeComponent();
        }

        private void frmMain_Load(object sender, EventArgs e)
        {
            // Instantiate a new sphere before setting properties
            Model.Globals.Sphere = new clsSphere();
            imageRendering = new ImageRendering();
            imageRendering.updateStatus += new ImageRendering.UpdateStatusDelegate(UpdateStatus);

            WindowState state = Initialisation.LoadSettings();

            this.WindowState = (state.State == "Normal" ? FormWindowState.Normal : FormWindowState.Maximized);

            if (state.Width > 0) this.Width = state.Width;
            if (state.Height > 0) this.Height = state.Height;
            if (state.Left > 0) this.Left = state.Left;
            if (state.Top > 0) this.Top = state.Top;

            string FilePath = Path.Combine(Globals.SetUp.NavPath, "home.json");
            LoadSphereFile(FilePath);
        }

        private void Main_FormClosed(object sender, FormClosedEventArgs e)
        {
            Initialisation.SaveSettings(this.Width, this.Height, this.Left, this.Top, this.WindowState);
        }

        public void mnuLoadSphere_Click(object sender, EventArgs e)
        {
            string FilePath = "";

            OpenFileDialog dlg = new OpenFileDialog();
            if (Directory.Exists(Globals.SetUp.NavPath))
            {
                dlg.InitialDirectory = Globals.SetUp.NavPath;
            }

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                FilePath = dlg.FileName;
            }

            picImage.Refresh();

            if (FilePath != "" && FilePath != null)
            {
                LoadSphereFile(FilePath);
            }
        }

        private void LoadSphereFile(string FilePath)
        {
            clsSphere sphere = Model.Globals.Sphere;

            // Extract address from pathname
            Globals.SetUp.NavPath = Path.GetDirectoryName(FilePath);

            if (Navigation.Navigate(FilePath))
            {
                if (sphere.ViewportImage == null)
                {  
                    if (sphere.RayMap != null)
                    {
                        RefreshImage();
                    }
                    else
                    {
                        RaytraceImage();
                    }
                }
                else
                {
                    picImage.Image = sphere.ViewportImage;
                }
            }
        }

        private void mnuRefresh_Click(object sender, EventArgs e)
        {
            RefreshImage();
        }

        private void mnuRender_Click(object sender, EventArgs e)
        {
            RaytraceImage();
        }

        private void RefreshImage()
        {
            if (Model.Globals.Sphere.RayMap != null)
            {
                staStatus.Items[0].Text = "Redisplaying...";
                Application.DoEvents();
                imageRendering.Redisplay();
                staStatus.Items[0].Text = "Completed";
                // Display the bitmap
                picImage.Image = Model.Globals.Sphere.ViewportImage;
            }
        }

        private void RaytraceImage()
        {
            imageRendering.InitialiseSphere();
            staStatus.Items[0].Text = "Raytracing started...";
            Application.DoEvents();
            imageRendering.PerformRayTracing();

            // Display the bitmap
            picImage.Image = Model.Globals.Sphere.ViewportImage;
        }

        /// <summary>
        /// Callback to update the status after a line has been processed.
        /// </summary>
        /// <param name="rowCount"></param>
        private void UpdateStatus(int rowCount, int totalLines)
        {
            string statusMessage = (rowCount < totalLines) ? rowCount + " of " + totalLines + " rows processed" : "Rendering completed";
            staStatus.Items[0].Text = statusMessage;
        }

        // Save current image
        private void mnuSaveImage_Click(object sender, EventArgs e)
        {
            string PathName = null;

            SaveFileDialog dlg = new SaveFileDialog();
            dlg.Filter =  "Jpeg files (*.jpg)|*.jpg|Tiff files (*.tif)|*.tif|" +
                          "Bitmap files (*.bmp)|*.bmp|Png files (*.png)|*.png";
            dlg.FileName = Path.GetFileNameWithoutExtension(currentAddress);

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                PathName = dlg.FileName;
                int filter = dlg.FilterIndex;
                ImageFormat format;

                switch (filter)
                {
                    case 2:
                        format = ImageFormat.Tiff;
                        break;
                    case 3:
                        format = ImageFormat.Bmp;
                        break;
                    case 4:
                        format = ImageFormat.Png;
                        break;
                    default:
                        format = ImageFormat.Jpeg;
                        break;
                }

                picImage.Image.Save(PathName, format);
            }
        }

        // TODO: Save sphere data
        private void mnuSaveSphere_Click(object sender, EventArgs e)
        {
            clsSphere sphere = Model.Globals.Sphere;

            string PathName = null;

            SaveFileDialog dlg = new SaveFileDialog();
            dlg.Filter = "Json files (*.json)|*.json";
            dlg.FileName = Path.GetFileNameWithoutExtension(currentAddress);

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                PathName = dlg.FileName;
                int filter = dlg.FilterIndex;

                // Save sphere as json data
                Navigation.SaveData(PathName);
            }
        }

        public void mnuClose_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void picImage_Resize(object sender, EventArgs e)
        {
            int ClientHeight = this.pnlImage.Height;
            int ClientWidth = this.pnlImage.Width;

            Globals.BitmapSizeType bitmapSize = getBitmapSize();
            float ImageRatio = (float)bitmapSize.Width / (float)bitmapSize.Height;

            if (bResizing) return;
            bResizing = true;

            // If the working area is wider format than the bitmap
            if ((float)ClientWidth / (float)ClientHeight > ImageRatio)
            {
                // Restrict the image height
                picImage.Height = ClientHeight;
                // Scale the image width
                picImage.Width = (int)(picImage.Height * ImageRatio);

                picImage.Top = 0;
                picImage.Left = (ClientWidth - picImage.Width) / 2;
            }
            else
            {
                // Restrict the image width
                picImage.Width = ClientWidth;
                // Scale the image height
                picImage.Height = (int)(picImage.Width / ImageRatio);

                picImage.Top = (ClientHeight - picImage.Height) / 2;
                picImage.Left = 0;
            }

            bResizing = false;
        }

        private Globals.BitmapSizeType getBitmapSize()
        {
            clsSphere sphere = Model.Globals.Sphere;
            double sphereResolution = sphere.AngularResolution * Globals.DEG_TO_RAD / 2;
            double stepSize = Math.Sin(sphereResolution);

            double verticalView = sphere.VerticalView * Globals.DEG_TO_RAD / 2;
            double maxVertical = Math.Sin(verticalView);

            double horizontalView = sphere.HorizontalView * Globals.DEG_TO_RAD / 2;
            double maxHorizontal = Math.Sin(horizontalView);

            Globals.BitmapSize.Height = (int)(maxVertical / stepSize);
            Globals.BitmapSize.Width = (int)(maxHorizontal / stepSize);

            return Globals.BitmapSize;
        }

        private int getBitmapWidth()
        {
            clsSphere sphere = Model.Globals.Sphere;
            double sphereResolution = sphere.AngularResolution * Globals.DEG_TO_RAD / 2;
            double stepSize = Math.Sin(sphereResolution);

            double horizontalView = sphere.HorizontalView * Globals.DEG_TO_RAD / 2;
            double maxHorizontal = Math.Sin(horizontalView);

            return (int)(maxHorizontal / stepSize);
        }

        private void mnuReset_Click(object sender, EventArgs e)
        {
        }

        private void Main_KeyUp(object sender, KeyEventArgs e)
        {
            bool bScaleChanged = false;
            double fWidth = 0, fHeight = 0;

            if (e.Control)
            {
                if (e.KeyCode == Keys.Add || e.KeyCode == Keys.Subtract)
                {
                    // Ctrl+ or Ctrl- was pressed (zoom image)
                    //double Scale = 1.25;
                    //if (e.KeyCode == Keys.Add)
                    //    Scale = 0.8;

                    //fWidth = (double)(DataClasses.Globals.Sphere.ImagePlane.Width * Scale);
                    //fHeight = (double)(DataClasses.Globals.Sphere.ImagePlane.Height * Scale);

                    //if (fWidth < DataClasses.Globals.Sphere.Resolution ||
                    //    fHeight < DataClasses.Globals.Sphere.Resolution)
                    //    return;

                    bScaleChanged = true;
                }
                else if (e.KeyCode == Keys.NumPad0 || e.KeyCode == Keys.D0)
                {
                    // Ctrl0 was pressed (full size image)
                    if (e.Alt)
                    {
                        double UnitsPerPixel = Model.Globals.Sphere.AngularResolution;
                        //fHeight = Globals.SetUp.BitmapHeight * UnitsPerPixel;
                       //fWidth = Globals.SetUp.BitmapWidth * UnitsPerPixel;
                    }
                    else
                    {
                        //float dRatio = (float)Globals.SetUp.BitmapWidth / Globals.SetUp.BitmapHeight;
                        //if (dRatio > 1)
                        //{
                        //    fHeight = 2;
                        //    fWidth = fHeight * dRatio;
                        //}
                        //else
                        //{
                        //    fWidth = 2;
                        //    fHeight = fWidth * dRatio;
                        //}
                    }

                    bScaleChanged = true;
                }

                if (bScaleChanged)
                {
                    double fTop = 0 - fHeight / 2;
                    double fLeft = 0 - fWidth / 2;
                    ResumeDisplay();
                }
            }
            else if (e.KeyCode == Keys.Left || e.KeyCode == Keys.Right || e.KeyCode == Keys.Up || e.KeyCode == Keys.Down)
            {
                // Update the latitude/longitude values
                SetViewingAngle(Model.Globals.Sphere.CentreLatitude, Model.Globals.Sphere.CentreLongitude);
                ResumeDisplay();
            }
        }

        // Redisplay the image using the current parameters
        private void ResumeDisplay()
        {
            if (m_DisplayStatus == DisplayStatus.InProgress)
                m_DisplayStatus = DisplayStatus.Stopped;

            tmrRedraw.Enabled = true;
        }

        private void tmrRedraw_Tick(object sender, EventArgs e)
        {
            if (m_DisplayStatus == DisplayStatus.None)
            {
                tmrRedraw.Enabled = false;
            }
        }

        private void mnuSphereSettings_Click(object sender, EventArgs e)
        {
            SphereSettings form = new SphereSettings();
            form.RefreshImage += new SphereSettings.RefreshDelegate(RefreshImage);
            form.ShowDialog(this);
        }

        private void mnuUserSettings_Click(object sender, EventArgs e)
        {
            UserSettings form = new UserSettings();
            form.ShowDialog(this);
        }

        private void mnuRotation_Click(object sender, EventArgs e)
        {
            Rotation form = new Rotation(Path.GetFileNameWithoutExtension(currentAddress));
            DialogResult result = form.ShowDialog(this);

            if (result == DialogResult.OK)
            {
                // Retrieve centre point coords
                double[] centreCoords = form.CentreCoords;

                // Record directory and base path for the sequence files
                string sequenceDirectory = Path.Combine(Globals.SetUp.SeqSource, form.BaseName);
                Directory.CreateDirectory(sequenceDirectory);

                string basePath = Path.Combine(sequenceDirectory, form.BaseName);
                // Determine the number of radians to turn per frame
                double[,] angles = form.Angles;
                // Retrieve the total frames
                int totalFrames = form.Stages;
                // Retrieve the file format
                ImageFormat format = form.Format;

                // Close the dialog
                form.Close();
                form.Dispose();

                // Call the sequence generator to perform the rotation and save the image and navigation files.
                Sequence sequence = new Sequence(imageRendering, picImage, format);
                sequence.PerformRotation(centreCoords, angles, totalFrames, basePath);
            }
        }

        private void mnuVideoConversion_Click(object sender, EventArgs e)
        {
            VideoConversion form = new VideoConversion();
            DialogResult result = form.ShowDialog(this);

            if (result == DialogResult.OK)
            {
                // Retrieve the directory for the sequence files
                string sequenceDirectory = form.sourceDirectory; ;
                // Retrieve the target filename
                string targetFile = form.targetFile;
                // Retrieve the frames per second
                int framesPerSecond = form.FramesPerSecond;
                int loops = form.Loops;

                // TODO: Read the first file and find the width and height
                int width = 360;
                int height = 360;

                // Close the dialog
                form.Close();
                form.Dispose();

                writeAVI(sequenceDirectory, targetFile, framesPerSecond, loops, width, height);
            }
        }

        private void writeAVI(string sequenceDirectory, string targetFile, int framesPerSecond, int loops, int width, int height)
        {
            // create instance of video writer
            VideoFileWriter writer = new VideoFileWriter();
            // create new video file
            writer.Open(targetFile, width, height, framesPerSecond, VideoCodec.MPEG4);
            // create a bitmap to save into the video file
            string[] files = Directory.GetFiles(sequenceDirectory, "*.*");
            Bitmap bitmap;

            for (int i = 0; i < loops; i++)
            {
                foreach (String filePath in files)
                {
                    bitmap = (Bitmap)Image.FromFile(filePath);
                    writer.WriteVideoFrame(bitmap);
                }
            }
            writer.Close();
        }

        private void settingsToolStripMenuItem_DropDownOpening(object sender, EventArgs e)
        {
            mnuRotation.Enabled = (imageRendering != null);
        }
    }
}