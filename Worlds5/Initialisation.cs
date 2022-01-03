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
    public struct WindowState {
        public int Width;
        public int Height;
        public int Left;
        public int Top;
        public string State;
    }

	sealed public class Initialisation 
	{ 
		//  Initialise settings from property settings
		public static WindowState LoadSettings() 
		{
            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string applicationPath = Path.Combine(appDataPath, "Worlds5");
            if (!Directory.Exists(applicationPath))
            {
                Directory.CreateDirectory(applicationPath);
            }
            string settingsPath = Path.Combine(applicationPath, "settings.json");
            WindowState windowState = new WindowState();

            if (!File.Exists(settingsPath))
            {
                // Save the default settings file to the app data path
                string sourceSettings = Path.Combine(Application.StartupPath, "default_settings.json");
                File.Copy(sourceSettings, settingsPath);
            }

            clsSphere sphere = Model.Globals.Sphere;
            SettingsData.RootObject settingsRoot = new SettingsData.RootObject();

            // Load the json settings file
            using (StreamReader r = new StreamReader(settingsPath))
            {
                string jsonSettings = r.ReadToEnd();
                settingsRoot = JsonConvert.DeserializeObject<SettingsData.RootObject>(jsonSettings);
            }

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

                // Main window
                windowState.State = mainWindow.MainState;
                windowState.Width = mainWindow.MainWidth;
                windowState.Height = mainWindow.MainHeight;
                windowState.Left = mainWindow.MainLeft;
                if (Screen.AllScreens.GetUpperBound(0) == 0 && windowState.Left > Screen.PrimaryScreen.Bounds.Width)
                {
                    windowState.Left = Screen.PrimaryScreen.Bounds.Width - windowState.Width;
                }
				windowState.Top = mainWindow.MainTop; 
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

            if (decodedSetting !=null) {
                decodedSetting = decodedSetting.Replace("%APPDATA%", appDataPath);
                decodedSetting = decodedSetting.Replace("%APPSTART%", Application.StartupPath);
            }
            return decodedSetting;
        }
        
		//  Save settings to property settings
		public static void SaveSettings(int windowWidth, int windowHeight, int windowLeft, int windowTop, FormWindowState windowState) 
		{
            SettingsData.Preferences prefs = new SettingsData.Preferences();
            SettingsData.Imaging imaging = new SettingsData.Imaging();
            SettingsData.MainWindow mainWindow = new SettingsData.MainWindow();

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

            SettingsData.RootObject settingsRoot = new SettingsData.RootObject();
            settingsRoot.Preferences = prefs;
            settingsRoot.Imaging = imaging;
            settingsRoot.MainWindow = mainWindow;

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string settingsPath = Path.Combine(appDataPath, "Worlds5", "settings.json");

            // Save the json file
            using (StreamWriter w = new StreamWriter(settingsPath))
            {
                string settingsJson = JsonConvert.SerializeObject(settingsRoot);
                w.Write(settingsJson);
            }
		} 
	} 
}
