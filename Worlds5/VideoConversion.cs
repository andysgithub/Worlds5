using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Text;
using System.Windows.Forms;

namespace Worlds5
{
    public partial class VideoConversion : Form
    {
        public VideoConversion()
        {
            InitializeComponent();
        }

        private void btnConvert_Click(object sender, EventArgs e)
        {
            SaveSettings();
            this.Hide();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        public string sourceDirectory
        {
            get { return txtSource.Text; }
        }

        public string targetFile
        {
            get { return txtTarget.Text; }
        }

        public int FramesPerSecond
        {
            get { return (int)updFramesPerSec.Value; }
        }

        public int Loops
        {
            get { return (int)updLoop.Value; }
        }

        private void btnSource_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog form = new FolderBrowserDialog();
            form.ShowNewFolderButton = true;
            form.SelectedPath = Globals.SetUp.SeqPath;

            DialogResult result = form.ShowDialog();

            if (result == DialogResult.OK)
            {
                // Get the source directory & write to the textbox
                txtSource.Text = form.SelectedPath;
            }
            form.Dispose();
        }

        private void btnTarget_Click(object sender, EventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Filter = "AVI files (*.avi)|*.avi";
            dlg.AddExtension = true;
            dlg.CheckFileExists = false;
            dlg.InitialDirectory = Globals.SetUp.SeqPath;

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                // Get the target file & write to the textbox
                txtTarget.Text = dlg.FileName;
                dlg.Dispose();
            }
        }

        private void LoadSettings()
        {
            Globals.SetUpType settings = Globals.SetUp;

            // File paths
            txtSource.Text = settings.SeqPath;
            txtTarget.Text = settings.SeqPath;

            // Sequence
            updFramesPerSec.Value = settings.FramesPerSec;
            updLoop.Value = settings.AutoRepeat;
        }

        private void SaveSettings()
        {
            // File paths
            Globals.SetUp.SeqPath = txtSource.Text;
            /*Globals.SetUp.SeqTarget = txtTarget.Text;*/

            // Sequence
            Globals.SetUp.FramesPerSec = (int)updFramesPerSec.Value;
            Globals.SetUp.AutoRepeat = (int)updLoop.Value;
        }

        private void VideoConversion_Load(object sender, EventArgs e)
        {
            LoadSettings();
        }
    }
}
