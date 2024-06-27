using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using Model;
using Accord.Video.FFMPEG;

namespace Worlds5
{
    public partial class Main : Form
    {
        private bool bResizing = false;
        private string currentAddress = string.Empty;
        private ImageRendering imageRendering = null;

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
            imageRendering.updateRowStatus += new ImageRendering.UpdateRowStatusDelegate(UpdateRowStatus);
            imageRendering.updateRayStatus += new ImageRendering.UpdateRayStatusDelegate(UpdateRayStatus);

            WindowState state = Initialisation.LoadSettings();

            this.WindowState = (state.State == "Normal" ? FormWindowState.Normal : FormWindowState.Maximized);

            if (state.Width > 0) this.Width = state.Width;
            if (state.Height > 0) this.Height = state.Height;
            if (state.Left > 0) this.Left = state.Left;
            if (state.Top > 0) this.Top = state.Top;

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string FilePath = Path.Combine(appDataPath, "Worlds5", "Navigation", "home.json");
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
                Model.Globals.CurrentAddress = FilePath;
            }
        }

        private async void LoadSphereFile(string FilePath)
        {
            clsSphere sphere = Model.Globals.Sphere;

            // Extract address from pathname
            Globals.SetUp.NavPath = Path.GetDirectoryName(FilePath);

            if (Navigation.Navigate(FilePath))
            {
                this.Text = Model.Globals.AppName + " - " + Path.GetFileNameWithoutExtension(FilePath);
                if (sphere.ViewportImage == null)
                {
                    if (sphere.RayMap != null)
                    {
                        RefreshImage();
                    }
                    else
                    {
                        await RaytraceImage();
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

        private async void mnuRaytrace_Click(object sender, EventArgs e)
        {

            await RaytraceImage();
        }

        private async void RefreshImage()
        {
            if (Model.Globals.Sphere.RayMap != null)
            {
                staStatus.Items[0].Text = "Redisplaying...";
                await imageRendering.Redisplay();
                staStatus.Items[0].Text = "Completed";
                // Display the bitmap
                picImage.Image = Model.Globals.Sphere.ViewportImage;
            }
            else
            {
                MessageBox.Show("You must perform raytracing before rendering.", "No Ray Map");
            }
        }

        private async Task<bool> RaytraceImage()
        {
            staStatus.Items[0].Text = "Initialising...";
            imageRendering.InitialiseSphere();
            if (staStatus != null && staStatus.Items != null && staStatus.Items.Count > 0)
            {
                staStatus.Items[0].Text = "Raytracing started...";
            }
            await imageRendering.PerformRayTracing();

            // Display the bitmap
            picImage.Image = Model.Globals.Sphere.ViewportImage;
            return true;
        }

        /// <summary>
        /// Callback to update the status after a line has been processed.
        /// </summary>
        /// <param name="rowCount"></param>
        private void UpdateRowStatus(int[] rowArray, int totalLines)
        {
            int rowCount = rowArray.Count(c => c == 1);
            if (staStatus != null && staStatus.Items != null && staStatus.Items.Count > 0)
            {
                string statusMessage = (rowCount < totalLines) ? rowCount + " of " + totalLines + " rows processed" : "Rendering completed";

                if (staStatus.InvokeRequired)
                {
                    staStatus.Invoke(new MethodInvoker(() => staStatus.Items[0].Text = statusMessage));
                }
                else
                {
                    staStatus.Items[0].Text = statusMessage;
                }
            }
        }

        /// <summary>
        /// Callback to update the status after a line has been processed.
        /// </summary>
        /// <param name="rayCount"></param>
        private void UpdateRayStatus(int[] rayArray, int totalRays)
        {
            int rayCount = rayArray.Count(c => c == 1);

            if (staStatus != null && staStatus.Items != null && staStatus.Items.Count > 0)
            {
                string statusMessage = (rayCount < totalRays) ? rayCount + " of " + totalRays + " rays processed" : "Rendering completed";

                if (staStatus.InvokeRequired)
                {
                    staStatus.Invoke(new MethodInvoker(() => staStatus.Items[0].Text = statusMessage));
                }
                else
                {
                    staStatus.Items[0].Text = statusMessage;
                }
            }
        }

        /// <summary>
        /// Callback to update the status with a given message.
        /// </summary>
        /// <param name="statusMessage"></param>
        private void UpdateStatus(string statusMessage)
        {
            if (staStatus != null && staStatus.Items != null && staStatus.Items.Count > 0)
            {
                try
                {
                    staStatus.Items[0].Text = statusMessage;
                }
                catch { }
            }
        }

        // Save current image
        private void mnuSaveImage_Click(object sender, EventArgs e)
        {
            SaveFileDialog dlg = new SaveFileDialog();
            dlg.Filter = "Jpeg files (*.jpg)|*.jpg|Tiff files (*.tif)|*.tif|" +
                          "Bitmap files (*.bmp)|*.bmp|Png files (*.png)|*.png";
            dlg.FileName = Path.GetFileNameWithoutExtension(currentAddress);

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                string PathName = dlg.FileName;
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
            double sphereResolution = sphere.settings.AngularResolution * Globals.DEG_TO_RAD / 2;
            double stepSize = Math.Sin(sphereResolution);

            double verticalView = sphere.settings.VerticalView * Globals.DEG_TO_RAD / 2;
            double maxVertical = Math.Sin(verticalView);

            double horizontalView = sphere.settings.HorizontalView * Globals.DEG_TO_RAD / 2;
            double maxHorizontal = Math.Sin(horizontalView);

            Globals.BitmapSize.Height = (int)(maxVertical / stepSize);
            Globals.BitmapSize.Width = (int)(maxHorizontal / stepSize);

            return Globals.BitmapSize;
        }

        private int getBitmapWidth()
        {
            clsSphere sphere = Model.Globals.Sphere;
            double sphereResolution = sphere.settings.AngularResolution * Globals.DEG_TO_RAD / 2;
            double stepSize = Math.Sin(sphereResolution);

            double horizontalView = sphere.settings.HorizontalView * Globals.DEG_TO_RAD / 2;
            double maxHorizontal = Math.Sin(horizontalView);

            return (int)(maxHorizontal / stepSize);
        }

        private void mnuReset_Click(object sender, EventArgs e)
        {
        }

        private void mnuSphereSettings_Click(object sender, EventArgs e)
        {
            SphereSettings form = new SphereSettings();
            form.RefreshImage += new SphereSettings.RefreshDelegate(RefreshImage);
            form.RaytraceImage += new SphereSettings.RaytraceDelegate(RaytraceImage);
            form.UpdateStatus += new SphereSettings.StatusDelegate(UpdateStatus);
            form.ShowDialog(this);
        }

        private void mnuUserSettings_Click(object sender, EventArgs e)
        {
            UserSettings form = new UserSettings();
            form.ShowDialog(this);
        }

        private void mnuRotation_Click(object sender, EventArgs e)
        {
            SequenceSettings form = new SequenceSettings(Path.GetFileNameWithoutExtension(currentAddress));
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

                // Retrieve the start and end sphere radius
                double[] sphereRadius = form.SphereRadius;

                // Retrieve the total frames
                int totalFrames = form.Stages;

                // Retrieve the file format
                ImageFormat format = form.Format;

                // Close the dialog
                form.Close();
                form.Dispose();

                // Call the sequence generator to perform the rotation and save the image and navigation files.
                Sequence sequence = new Sequence(imageRendering, picImage, format);
                sequence.PerformRotation(centreCoords, angles, sphereRadius, totalFrames, basePath);
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

                // Close the dialog
                form.Close();
                form.Dispose();

                writeAVI(sequenceDirectory, targetFile, framesPerSecond, loops);
            }
        }

        private void writeAVI(string sequenceDirectory, string targetFile, int framesPerSecond, int loops)
        {
            VideoFileWriter writer = new VideoFileWriter();

            // Read the first file to find the width and height
            string[] files = Directory.GetFiles(sequenceDirectory, "*.*");
            Image img = Image.FromFile(files[0]);
            // Make the size values even
            int width = img.Width - img.Width % 2;
            int height = img.Height - img.Height % 2;
            // Set the bitrate in megabits per second
            float Mbps = 30.0f;

            // Open a new video file for this image size
            writer.Open(targetFile, width, height, framesPerSecond, VideoCodec.MPEG4, (int)(Mbps * 1000000));

            // Create a bitmap to save into the video file
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