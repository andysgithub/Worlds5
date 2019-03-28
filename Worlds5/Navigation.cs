using System;
using System.IO;
using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Diagnostics;
using System.Windows.Forms;
using Model;
using Newtonsoft.Json;
using System.IO.Compression;

namespace Worlds5
{
    public static class Navigation
    {
        public static string LocationName;

        public static bool Navigate(string Address)
        {
            // Check that address is available if attempting to load file
            if (Address != "")
            {
                // Load navigation file from Address if known
                if (File.Exists(Address))
                {
                    // Load navigation parameters
                    if (LoadData(Address))
                    {
                        return true;
                    }
                }
                else
                {
                    MessageBox.Show("The file is not available:\n" + Address, "Navigation Error 2",
                                    MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                }
            }
            return false;
        }

        //  Read navigation data into RenderImage class
        private static bool LoadData(string spherePath)
        {
            try
            {
                clsSphere sphere = Model.Globals.Sphere;
                SphereData.RootObject sphereRoot = new SphereData.RootObject();

                // Load the json sphere file
                using (StreamReader r = new StreamReader(spherePath))
                {
                    string jsonSettings = r.ReadToEnd();
                    sphereRoot = JsonConvert.DeserializeObject<SphereData.RootObject>(jsonSettings);
                }

                SphereData.Type fileInfo = sphereRoot.Type;
                SphereData.Navigation navigation = sphereRoot.Navigation;
                SphereData.Viewing viewing = sphereRoot.Viewing;
                SphereData.Raytracing raytracing = sphereRoot.Raytracing;
                SphereData.Rendering rendering = sphereRoot.Rendering;

                // Load file-type reference and total dimensions
                int FileType = fileInfo.FileType;

                // Redimension the transformation matrix
                int dimensions = Convert.ToInt16(fileInfo.Dimensions);
                Model.Globals.Dimensions = dimensions;

                // Load Navigation settings
                sphere.PositionMatrix = navigation.PositionMatrix;

                // Calculate size of ray map
                int raysX = (int)(viewing.HorizontalView / viewing.AngularResolution) + 1;
                int raysY = (int)(viewing.VerticalView / viewing.AngularResolution) + 1;
                // Convert base64 string to compressed byte array
                byte[] compressedData = Convert.FromBase64String(navigation.RayMap);
                // Uncompress byte array
                byte[] rawData = Decompress(compressedData);
                // Initialise the ray map size to correspond to the viewport
                sphere.RayMap = new TracedRay.RayDataType[raysX, raysY];
                // Copy raw data to the ray map
                for(int x = 0; x < raysX; x++) {
                    for(int x = 0; x < raysX; x++) {
                        sphere.RayMap[x,y] = ((TracedRay.RayDataType*)rawData)[i++];
                    }
                }

                // Convert base64 string to compressed byte array
                byte[] compressedData = Convert.FromBase64String(navigation.ViewportImage);
                // Uncompress byte array
                byte[] imageData = Decompress(compressedData);
                // Convert byte array to image
                if (imageData.Length > 0)
                {
                    sphere.ViewportImage = (Bitmap)Image.FromStream(new MemoryStream(imageData));
                }

                // Load Rendering settings
                if (FileType == 3)
                {
                    ImageRendering.Bailout = raytracing.Bailout;

                    // Viewing window
                    sphere.AngularResolution = viewing.AngularResolution;
                    sphere.Radius = viewing.Radius;
                    sphere.CentreLatitude = viewing.CentreLatitude;
                    sphere.CentreLongitude = viewing.CentreLongitude;
                    sphere.VerticalView = viewing.VerticalView;
                    sphere.HorizontalView = viewing.HorizontalView;

                    // Raytracing
                    sphere.SamplingInterval = raytracing.SamplingInterval;
                    sphere.SurfaceThickness = raytracing.SurfaceThickness;
                    sphere.RayPoints = raytracing.RayPoints;
                    sphere.MaxSamples = raytracing.MaxSamples;
                    sphere.BoundaryInterval = raytracing.BoundaryInterval;
                    sphere.BinarySearchSteps = raytracing.BinarySearchSteps;
                    sphere.ActiveIndex = raytracing.ActiveIndex;

                    // Rendering
                    sphere.ExposureValue = rendering.ExposureValue;
                    sphere.Saturation = rendering.Saturation;
                    sphere.StartDistance = rendering.StartDistance;
                    sphere.EndDistance = rendering.EndDistance;
                    sphere.ColourDetail = rendering.ColourDetail;
                    sphere.SurfaceContrast = rendering.SurfaceContrast;
                    sphere.LightingAngle = rendering.LightingAngle;
                }
                else
                {
                    MessageBox.Show(
                        "File type not recognised:\n" + spherePath + "\n",
                        "Navigation Error",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Exclamation);
                    return false;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Error reading navigation file:\n" + spherePath + "\n" + ex.Message,
                    "Navigation Error",
                    MessageBoxButtons.RetryCancel,
                    MessageBoxIcon.Exclamation);
                return false;
            }
            return true;
        }

        public static bool SaveData(string spherePath)
        {
            float[] ColourDetail = new float[2];

            clsSphere sphere = Model.Globals.Sphere;

            SphereData.Type fileInfo = new SphereData.Type();
            SphereData.Navigation navigation = new SphereData.Navigation();
            SphereData.Viewing viewing = new SphereData.Viewing();
            SphereData.Raytracing raytracing = new SphereData.Raytracing();
            SphereData.Rendering rendering = new SphereData.Rendering();

            try
            {
                fileInfo.FileType = 3;

                int dimensions = Model.Globals.Dimensions;

                navigation.PositionMatrix = sphere.PositionMatrix;

/////////////////////////////////////////////////////////

                TracedRay.RayDataType[,] rayMap = sphere.RayMap;

                // Size a byte array to store the ray map data
                int structSize = Marshal.SizeOf(typeof(TracedRay.RayDataType));
                List<byte> rawData = new List<byte>();

                // Copy the ray map data to the byte array
                for(int x = 0; x < sphere.RayMap.GetLength(0); x++) {
                    for(int y = 0; y < sphere.RayMap.GetLength(1); y++) {
                        rawData.Add(sphere.RayMap[x,y]);
                    }
                }

                // // Obtain a GCHandle instance
                // gchnd = GCHandle.Alloc(rayMap, GCHandleType.Pinned);
                // // Copy to the byte array
                // Marshal.Copy(gchnd.AddrOfPinnedObject, rawData, 0, rawData.Length);

                // Compress byte array
                byte[] compressedData = Compress(rawData.ToArray());
                // Convert compressed data to base64 string
                navigation.RayMap = Convert.ToBase64String(compressedData);

/////////////////////////////////////////////////////////

                ImageConverter converter = new ImageConverter();
                // Convert image to byte array
                byte[] imageData = (byte[])converter.ConvertTo(sphere.ViewportImage, typeof(byte[]));
                // Compress byte array
                byte[] compressedData = Compress(imageData);
                // Convert compressed data to base64 string
                navigation.ViewportImage = Convert.ToBase64String(compressedData);

                raytracing.Bailout = ImageRendering.Bailout;

                // Viewing window
                viewing.AngularResolution = sphere.AngularResolution;
                viewing.Radius = sphere.Radius;
                viewing.CentreLatitude = sphere.CentreLatitude;
                viewing.CentreLongitude = sphere.CentreLongitude;
                viewing.VerticalView = sphere.VerticalView;
                viewing.HorizontalView = sphere.HorizontalView;

                // Raytracing
                raytracing.SamplingInterval = sphere.SamplingInterval;
                raytracing.SurfaceThickness = sphere.SurfaceThickness;
                raytracing.RayPoints = sphere.RayPoints;
                raytracing.MaxSamples = sphere.MaxSamples;
                raytracing.BoundaryInterval = sphere.BoundaryInterval;
                raytracing.BinarySearchSteps = sphere.BinarySearchSteps;
                raytracing.ActiveIndex = sphere.ActiveIndex;

                // Rendering
                rendering.ExposureValue = sphere.ExposureValue;
                rendering.Saturation = sphere.Saturation;
                rendering.StartDistance = sphere.StartDistance;
                rendering.EndDistance = sphere.EndDistance;
                rendering.ColourDetail = sphere.ColourDetail;
                rendering.SurfaceContrast = sphere.SurfaceContrast;
                rendering.LightingAngle = sphere.LightingAngle;

                SphereData.RootObject sphereRoot = new SphereData.RootObject();
                sphereRoot.Type = fileInfo;
                sphereRoot.Navigation = navigation;
                sphereRoot.Viewing = viewing;
                sphereRoot.Raytracing = raytracing;
                sphereRoot.Rendering = rendering;

                // Save the json file
                using (StreamWriter w = new StreamWriter(spherePath))
                {
                    string jsonSettings = JsonConvert.SerializeObject(sphereRoot);
                    w.Write(jsonSettings);

                }
            }
            catch (Exception ex)
            {
                int ErrorNo = (int)(
                    MessageBox.Show(
                        "Couldn't write data to the file:\n" + spherePath + "\n\nError reported: " + ex.Message + "\n",
                        "Save Data Error",
                        MessageBoxButtons.RetryCancel,
                        MessageBoxIcon.Exclamation));

                if (ErrorNo == (int)DialogResult.Cancel)
                {
                    return false;
                }
            }
            return true;
        }

        public static byte[] Compress(byte[] data)
        {
            MemoryStream output = new MemoryStream();
            using (DeflateStream dstream = new DeflateStream(output, CompressionLevel.Optimal))
            {
                dstream.Write(data, 0, data.Length);
            }
            return output.ToArray();
        }

        public static byte[] Decompress(byte[] data)
        {
            MemoryStream input = new MemoryStream(data);
            MemoryStream output = new MemoryStream();
            using (DeflateStream dstream = new DeflateStream(input, CompressionMode.Decompress))
            {
                dstream.CopyTo(output);
            }
            return output.ToArray();
        }
    }
}
