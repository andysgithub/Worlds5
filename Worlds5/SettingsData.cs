﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Worlds5
{
    public class SettingsData
    {
        public class MainWindow
        {
            public string MainState { get; set; }
            public int MainWidth { get; set; }
            public int MainHeight { get; set; }
            public int MainLeft { get; set; }
            public int MainTop { get; set; }
        }

        public class User
        {
            public string NavPath { get; set; }
            public string SeqPath { get; set; }
            public bool Toolbar { get; set; }
            public bool Labels { get; set; }
            public bool ToolTips { get; set; }
            public bool StatusBar { get; set; }
        }

        public class Imaging
        {
            public int FramesPerSec { get; set; }
            public bool AutoRepeat { get; set; }
            public int BitmapHeight { get; set; }
            public int BitmapWidth { get; set; }
        }

        public class RootObject
        {
            public MainWindow MainWindow { get; set; }
            public User User { get; set; }
            public Imaging Imaging { get; set; }
        }
    }
}
