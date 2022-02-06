using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Worlds5
{
    public partial class UserSettings : Form
    {
        public UserSettings()
        {
            InitializeComponent();
        }

        private void UserSettings_Load(object sender, EventArgs e)
        {
            LoadSettings();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            SaveSettings();
            this.Close();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void btnNavigationPath_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog form = new FolderBrowserDialog();
            form.ShowNewFolderButton = true;
            form.SelectedPath = Globals.SetUp.NavPath;

            DialogResult result = form.ShowDialog();

            if (result == DialogResult.OK)
            {
                // Get the source directory & write to the textbox
                txtNavigationPath.Text = form.SelectedPath;
            }
            form.Dispose();
        }

        private void btnSequencePath_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog form = new FolderBrowserDialog();
            form.ShowNewFolderButton = true;
            form.SelectedPath = Globals.SetUp.SeqPath;

            DialogResult result = form.ShowDialog();

            if (result == DialogResult.OK)
            {
                // Get the source directory & write to the textbox
                txtSequencePath.Text = form.SelectedPath;
            }
            form.Dispose();
        }

        private void LoadSettings()
        {
            Globals.SetUpType settings = Globals.SetUp;

            // File paths
            txtNavigationPath.Text = settings.NavPath;
            txtSequencePath.Text = settings.SeqPath;

            // Main window
            chkToolbar.Checked = settings.Toolbar;
            chkLabels.Checked = settings.Labels;
            chkTooltips.Checked = settings.ToolTips;
            chkStatusBar.Checked = settings.StatusBar;
        }

        private void SaveSettings()
        {
            // File paths
            Globals.SetUp.NavPath = txtNavigationPath.Text;
            Globals.SetUp.SeqPath = txtSequencePath.Text;

            // Main window
            Globals.SetUp.Toolbar = chkToolbar.Checked;
            Globals.SetUp.Labels = chkLabels.Checked;
            Globals.SetUp.ToolTips = chkTooltips.Checked;
            Globals.SetUp.StatusBar = chkStatusBar.Checked;
        }
    }
}
