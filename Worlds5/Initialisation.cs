using System;
using System.Collections;
using System.Configuration;
using System.Data;
using System.Drawing;
using System.IO;
using System.Diagnostics;
using System.Windows.Forms;
using Microsoft.Win32;
using Model;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace Worlds5 
{
    private struct WindowState {
        private int width;
        private int height;
        private int left;
        private int top;
        private string state;
    }

	sealed public class Initialisation 
	{ 
		//  Initialise settings from property settings
		public static WindowState LoadSettings() 
		{
            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string settingsPath = Path.Combine(appDataPath, "Worlds5", "settings.json");
            WindowState windowState;

            if (!File.Exists(settingsPath))
            {
                // Save the default settings file to the app data path
                string sourceSettings = Path.Combine(Application.StartupPath, "default_settings.json");
                File.Copy(sourceSettings, settingsPath);
            }

            SettingsData settingsData = new SettingsData();

            // Load the json settings file
            using (StreamReader r = new StreamReader(settingsPath))
            {
                string jsonSettings = r.ReadToEnd();
                settingsData = JsonConvert.DeserializeObject<SettingsData>(jsonSettings);
            }

            clsSphere sphere = Model.Globals.Sphere;
            SettingsData.RootObject settingsRoot = new settingsData.RootObject();

            SettingsData.User user = settingsRoot.User;
            SettingsData.Imaging imaging = settingsRoot.Imaging;
            SettingsData.MainWindow mainWindow = settingsRoot.MainWindow;

            try
            {
                // User settings
                Globals.SetUp.NavPath = DecodeTag(user.NavPath);
                Globals.SetUp.SeqPath = DecodeTag(user.SeqPath);
                Globals.SetUp.Toolbar = user.Toolbar;
                Globals.SetUp.Labels = user.Labels; 
                Globals.SetUp.ToolTips = user.ToolTips;
                Globals.SetUp.StatusBar = user.StatusBar; 

                // Imaging settings
				Globals.SetUp.FramesPerSec = imaging.FramesPerSec;
                Globals.SetUp.AutoRepeat = imaging.AutoRepeat;
                Globals.SetUp.BitmapWidth = imaging.BitmapWidth;
                Globals.SetUp.BitmapHeight = imaging.BitmapHeight;

                windowState.state = mainWindow.MainState;
                windowState.width = mainWindow.MainWidth;
                windowState.height = mainWindow.MainHeight;
                windowState.left = mainWindow.MainLeft;
                if (Screen.AllScreens.GetUpperBound(0) == 0 && iLeft > Screen.PrimaryScreen.Bounds.Width)
                {
                    iLeft = Screen.PrimaryScreen.Bounds.Width - iWidth;
                }
				windowState.top = mainWindow.MainTop; 
			}
			catch  
			{ 
			}
            return windowState; 
		}

        private static string DecodeTag(string setting)
        {
            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string decodedSetting = setting;

            decodedSetting = decodedSetting.Replace("%APPDATA%", appDataPath);
            decodedSetting = decodedSetting.Replace("%APPSTART%", Application.StartupPath);
            return decodedSetting;
        }
        
		//  Save settings to property settings
		public static void SaveSettings(int iWidth, int iHeight, int iLeft, int iTop, FormWindowState fwsState) 
		{
            clsSphere sphere = Model.Globals.Sphere;
            SettingsData settingsData = new SettingsData();
            SettingsData.RootObject settingsRoot = new settingsData.RootObject();

            SettingsData.User user = settingsRoot.User;
            SettingsData.Imaging imaging = settingsRoot.Imaging;
            SettingsData.MainWindow mainWindow = settingsRoot.MainWindow;

            try
			{
                user.NavPath = Globals.SetUp.NavPath;
                user.SeqPath = Globals.SetUp.SeqPath;

                user.Toolbar = Globals.SetUp.Toolbar;
                user.Labels = Globals.SetUp.Labels;
                user.ToolTips = Globals.SetUp.ToolTips;
                user.StatusBar = Globals.SetUp.StatusBar;

                // Imaging settings
                imaging.FramesPerSec = Globals.SetUp.FramesPerSec;
                imaging.AutoRepeat = Globals.SetUp.AutoRepeat;
                imaging.BitmapWidth = Globals.SetUp.BitmapWidth;
                imaging.BitmapHeight = Globals.SetUp.BitmapHeight;

                if (fwsState != FormWindowState.Minimized) 
				{
                    mainWindow.MainState = fwsState.ToString();
					if (fwsState != FormWindowState.Maximized) 
					{
                        mainWindow.MainWidth = iWidth;
                        mainWindow.MainHeight = iHeight;
                        mainWindow.MainLeft = iLeft;
                        mainWindow.MainTop = iTop;
					} 
				} 
			}
			catch
			{
			}

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string settingsPath = Path.Combine(appDataPath, "Worlds5", "settings.json");

            // Save the json file
            using (StreamWriter w = new StreamWriter(settingsPath))
            {
                string settingsJson = JsonConvert.SerializeObject<SettingsData>(jsonSettings);
                w.Write(settingsJson);
                
            }
		} 
	} 
}
