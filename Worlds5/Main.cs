using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Model;
using AviFile;

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
        private int iWidth = 0, iHeight = 0, iLeft = 0, iTop = 0;
        private string sState = "Normal";
        private string currentAddress = string.Empty;
        private ImageRendering imageRendering = null;

        [DllImport("Unmanaged.dll")]
        static extern void InitBitmap(int Width, int Height);
        [DllImport("Unmanaged.dll")]
        static extern void NewImagePlane(RectangleF ImagePlane);
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
            string VersionNumber = FileVersionInfo.GetVersionInfo
                                                (System.Reflection.Assembly.GetExecutingAssembly().Location)
                                            .FileMajorPart.ToString().Trim() + "." +
                                   FileVersionInfo.GetVersionInfo
                                                (System.Reflection.Assembly.GetExecutingAssembly().Location)
                                            .FileMinorPart.ToString().Trim();
            
            // Instantiate a new sphere before setting properties
            Model.Globals.Sphere = new clsSphere();
            imageRendering = new ImageRendering();
            imageRendering.updateStatus += new ImageRendering.UpdateStatusDelegate(UpdateStatus);

            Initialisation.LoadSettings(ref iWidth, ref iHeight, ref iLeft, ref iTop, ref sState);
            
            if (sState == "Normal")
            {
                this.WindowState = FormWindowState.Normal;
            }
            else
            {
                this.WindowState = FormWindowState.Maximized;
            }

            if (iWidth > 0) this.Width = iWidth;
            if (iHeight > 0) this.Height = iHeight;
            if (iLeft > 0) this.Left = iLeft;
            if (iTop > 0) this.Top = iTop;
        }

        private void Main_FormClosed(object sender, FormClosedEventArgs e)
        {
            Initialisation.SaveSettings(this.Width, this.Height, this.Left, this.Top, this.WindowState);
        }

        public void mnuLoad_Click(object sender, EventArgs e)
        {
            string PathName = null;

            OpenFileDialog dlg = new OpenFileDialog();

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                PathName = dlg.FileName;
            }

            picImage.Refresh();

            if (PathName != "" && PathName != null)
            {
                // Extract address from pathname
                Globals.SetUp.NavPath = Path.GetDirectoryName(PathName);
                currentAddress = Globals.SetUp.NavPath + "\\" + Path.GetFileNameWithoutExtension(PathName);

                if (Navigation.Navigate(currentAddress))
                {
                    if (File.Exists(currentAddress + ".jpg"))
                    {
                        // Display the existing image for this location
                        picImage.Image = Image.FromFile(currentAddress + ".jpg");
                    }
                    else if (File.Exists(currentAddress + ".png"))
                    {
                        // Display the existing image for this location
                        picImage.Image = Image.FromFile(currentAddress + ".png");
                    }
                    else
                    {
                        imageRendering.InitialiseSphere();
                        imageRendering.PerformRayTracing();

                        // Display the bitmap
                        picImage.Image = imageRendering.GetBitmap();
                    }
                }
            }
        }

        /// <summary>
        /// Callback to update the image from a line processing node.
        /// </summary>
        /// <param name="latitude"></param>
        /// <param name="longitude"></param>
        /// <param name="rayColors"></param>
        //private void GetRayData(double latitude, double longitude, Model.Globals.RGBQUAD rayColors)
        //{
        //    imageDisplay.updateImage(latitude, longitude, rayColors);
        //}

        /// <summary>
        /// Callback to update the status after a line has been processed.
        /// </summary>
        /// <param name="rowCount"></param>
        private void UpdateStatus(int rowCount)
        {
            staStatus.Items[0].Text = "Rows processed: " + rowCount;
        }

        private void mnuSave_Click(object sender, EventArgs e)
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

        private void mnuRefresh_Click(object sender, EventArgs e)
        {
            imageRendering.InitialiseSphere();
            imageRendering.Redisplay();
        }

        public void mnuClose_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void picImage_Resize(object sender, EventArgs e)
        {
            int ClientHeight = this.pnlImage.Height;
            int ClientWidth = this.pnlImage.Width;
            int BitmapWidth = Globals.SetUp.BitmapWidth;
            int BitmapHeight = Globals.SetUp.BitmapHeight;
            float ImageFormat = (float)BitmapWidth / (float)BitmapHeight;

            if (bResizing) return;
            bResizing = true;

            // If the working area is wider format than the bitmap
            if ((float)ClientWidth / (float)ClientHeight > ImageFormat)
            {
                // Restrict the image height
                picImage.Height = ClientHeight;
                // Scale the image width
                picImage.Width = (int)(picImage.Height * ImageFormat);

                picImage.Top = 0;
                picImage.Left = (ClientWidth - picImage.Width) / 2;
            }
            else
            {
                // Restrict the image width
                picImage.Width = ClientWidth;
                // Scale the image height
                picImage.Height = (int)(picImage.Width / ImageFormat);

                picImage.Top = (ClientHeight - picImage.Height) / 2;
                picImage.Left = 0;
            }

            bResizing = false;
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
                        fHeight = Globals.SetUp.BitmapHeight * UnitsPerPixel;
                        fWidth = Globals.SetUp.BitmapWidth * UnitsPerPixel;
                    }
                    else
                    {
                        float dRatio = (float)Globals.SetUp.BitmapWidth / Globals.SetUp.BitmapHeight;
                        if (dRatio > 1)
                        {
                            fHeight = 2;
                            fWidth = fHeight * dRatio;
                        }
                        else
                        {
                            fWidth = 2;
                            fHeight = fWidth * dRatio;
                        }
                    }

                    bScaleChanged = true;
                }

                if (bScaleChanged)
                {
                    double fTop = 0 - fHeight / 2;
                    double fLeft = 0 - fWidth / 2;

                    //DataClasses.Globals.Sphere.ImagePlane = new RectangleF((float)fLeft, (float)fTop, (float)fWidth, (float)fHeight);
                    //NewImagePlane(DataClasses.Globals.Sphere.ImagePlane);

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

        private void mnuSettings_Click(object sender, EventArgs e)
        {
            Settings form = new Settings();
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
                string sequenceDirectory = Path.Combine(Globals.SetUp.SeqPath, form.BaseName);
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
                // Retrieve the file format
                string extension = form.Extension;

                // Close the dialog
                form.Close();
                form.Dispose();

                writeAVI(sequenceDirectory, targetFile, framesPerSecond, extension, true);
            }
        }

        private void writeAVI(string sequenceDirectory, string targetFile, int framesPerSecond, string extension, bool compress)
        {
            string sourceDirectory = sequenceDirectory;
            string[] files = Directory.GetFiles(sourceDirectory, "*." + extension);
            AviManager aviManager = new AviManager(targetFile, false);
            bool firstFrame = true;
            Bitmap bitmap;
            VideoStream aviStream = null;

            for (int loop = 0; loop < 5; loop++)
            {
                foreach (String filePath in files)
                {
                    bitmap = (Bitmap)Image.FromFile(filePath);
                    if (firstFrame)
                    {
                        aviStream = aviManager.AddVideoStream(compress, framesPerSecond, bitmap);
                        firstFrame = false;
                    }
                    else
                    {
                        aviStream.AddFrame(bitmap);
                        bitmap.Dispose();
                    }
                }
            }
            aviManager.Close();
        }

        private void settingsToolStripMenuItem_DropDownOpening(object sender, EventArgs e)
        {
            mnuRotation.Enabled = (imageRendering != null);
        }

    }
}