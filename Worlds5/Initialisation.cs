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

            SettingsData.Preferences prefs = settingsRoot.Preferences;
            SettingsData.Imaging imaging = settingsRoot.Imaging;
            SettingsData.MainWindow mainWindow = settingsRoot.MainWindow;

            try
            {
                // Preferences
                Globals.SetUp.NavPath = DecodeTag(prefs.NavPath);
                Globals.SetUp.SeqPath = DecodeTag(prefs.SeqPath);
                Globals.SetUp.Toolbar = prefs.Toolbar;
                Globals.SetUp.Labels = prefs.Labels; 
                Globals.SetUp.ToolTips = prefs.ToolTips;
                Globals.SetUp.StatusBar = prefs.StatusBar; 

                // Imaging settings
				Globals.SetUp.FramesPerSec = imaging.FramesPerSec;
                Globals.SetUp.AutoRepeat = imaging.AutoRepeat;
                Globals.SetUp.BitmapWidth = imaging.BitmapWidth;
                Globals.SetUp.BitmapHeight = imaging.BitmapHeight;

                // Main window
                windowState.state = mainWindow.MainState;
                windowState.width = mainWindow.MainWidth;
                windowState.height = mainWindow.MainHeight;
                windowState.left = mainWindow.MainLeft;
                if (Screen.AllScreens.GetUpperBound(0) == 0 && windowLeft > Screen.PrimaryScreen.Bounds.Width)
                {
                    windowLeft = Screen.PrimaryScreen.Bounds.Width - windowWidth;
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
		public static void SaveSettings(int windowWidth, int windowHeight, int windowLeft, int windowTop, FormWindowState windowState) 
		{
            clsSphere sphere = Model.Globals.Sphere;
            SettingsData settingsData = new SettingsData();
            SettingsData.RootObject settingsRoot = new settingsData.RootObject();

            SettingsData.Preferences prefs = settingsRoot.Preferences;
            SettingsData.Imaging imaging = settingsRoot.Imaging;
            SettingsData.MainWindow mainWindow = settingsRoot.MainWindow;

            try
			{
                // Preferences
                prefs.NavPath = Globals.SetUp.NavPath;
                prefs.SeqPath = Globals.SetUp.SeqPath;
                prefs.Toolbar = Globals.SetUp.Toolbar;
                prefs.Labels = Globals.SetUp.Labels;
                prefs.ToolTips = Globals.SetUp.ToolTips;
                prefs.StatusBar = Globals.SetUp.StatusBar;

                // Imaging settings
                imaging.FramesPerSec = Globals.SetUp.FramesPerSec;
                imaging.AutoRepeat = Globals.SetUp.AutoRepeat;
                imaging.BitmapWidth = Globals.SetUp.BitmapWidth;
                imaging.BitmapHeight = Globals.SetUp.BitmapHeight;

                // Main window
                if (windowState != FormWindowState.Minimized) 
				{
                    mainWindow.MainState = windowState.ToString();
					if (windowState != FormWindowState.Maximized) 
					{
                        mainWindow.MainWidth = windowWidth;
                        mainWindow.MainHeight = windowHeight;
                        mainWindow.MainLeft = windowLeft;
                        mainWindow.MainTop = windowTop;
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
